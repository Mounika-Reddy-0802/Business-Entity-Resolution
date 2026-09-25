"""Candidate generation at full scale: inverted-index hash joins -> data/candidates/{split}_candidates.parquet.

Business names repeat across many different entities in this data (86 Source 1 businesses in the
US are called "Blue Hypnosis"), so a name alone rarely identifies a business; name plus location
does. The noise is mostly per-token (typos, transliteration, moved suffixes, reordered address
components), so keys combine name skeleton tokens with single address words or numbers. Every key
is hashed to a 64-bit integer together with the country (country is only ever an equality
constraint), both sides are exploded to (key, record) rows, keys shared by more than MAX_BLOCK
Source 2/3 records are skipped, and the join gives the candidate pairs. Candidates are ranked by a
cheap skeleton similarity and capped.

Keys:
  kp   a pair of name skeleton tokens (DBA alternative name included), looser block limit:
       the only key for records without an address
  kn   the full name skeleton + one address word
  knp  a pair of name skeleton tokens + one address number
  kt   one name skeleton token (>= 3 letters) + one address word: survives a typo in the others
  ka   an address number + one address word (renamed or DBA records)

Writes s1_id, cand_id, kp, kn, knp, kt, ka, name_sim, addr_sim, cheap_score.

    python -m src.blocking.block train test [--report [--log <tag> "<change>"]]
"""
import os
import sys
from itertools import combinations
from multiprocessing import Pool

import numpy as np
import pandas as pd
from rapidfuzz import fuzz, process

from ..common.evaluate import log_run
from ..common.io_utils import DATA, SOURCES, is_fresh, load_ground_truth
from .normalise import load_normalised, skeleton

KEYS = ["kp", "kn", "knp", "kt", "ka"]
MAX_BLOCK = {"kp": 150, "kn": 60, "knp": 60, "kt": 60, "ka": 60}
MAX_TOKENS = 4            # name tokens used for pair keys (first 4 in sorted order: <= 6 pairs)
CAP = 40                  # final candidates per S1 entity
REPORT_CAPS = (10, 20, 25, 40, 60, 80)
S1_CHUNK = 250_000
COLS = ["entity_id", "country", "name_skel", "name_alt", "addr_numbers", "addr_skel"]
PART = 100_000
WORKERS = 10


def name_tokens(skel, alt):
    """Distinct skeleton tokens of the name and of its DBA alternative (>= 2 chars)."""
    toks = {t for t in skel.split() if len(t) >= 2}
    if alt:
        toks |= {t for t in skeleton(alt).split() if len(t) >= 2}
    return sorted(toks)


def address_words(askel):
    """Up to 6 address skeleton words (>= 2 letters), longest first: the street and city words."""
    words = {w for w in askel.split() if len(w) >= 2 and not w.isdigit()}
    return sorted(words, key=lambda w: (-len(w), w))[:6]


def address_numbers(nums):
    """Up to 3 distinct address numbers, in address order (single digits count: "C-1", "4/1")."""
    return list(dict.fromkeys(nums.split()))[:3]


def record_keys(df, kind):
    """(key string, row position) pairs of one key kind for a chunk of records."""
    out = []
    rows = zip(df.country, df.name_skel, df.name_alt, df.addr_numbers, df.addr_skel)
    for pos, (c, skel, alt, nums, askel) in enumerate(rows):
        toks = name_tokens(skel, alt)
        pairs = [f"{a}|{b}" for a, b in combinations(toks[:MAX_TOKENS], 2)] or toks
        if kind == "kp":
            out += [(f"{c}|{p}", pos) for p in pairs]
        elif kind == "kn":
            out += [(f"{c}|{skel}|{w}", pos) for w in address_words(askel)]
        elif kind == "knp":
            out += [(f"{c}|{p}|{n}", pos) for p in pairs for n in address_numbers(nums)]
        elif kind == "kt":
            out += [(f"{c}|{t}|{w}", pos) for t in toks[:MAX_TOKENS] if len(t) >= 3
                    for w in address_words(askel)]
        elif kind == "ka":
            out += [(f"{c}|{n}|{w}", pos) for n in address_numbers(nums) for w in address_words(askel)]
    return out


def _keys_chunk(args):
    df, offset, kind = args
    rows = record_keys(df, kind)
    key = pd.util.hash_array(np.array([r[0] for r in rows], dtype=object))
    idx = np.fromiter((r[1] + offset for r in rows), dtype=np.int64, count=len(rows))
    return key, idx


def exploded_keys(df, kind):
    """DataFrame key (uint64 hash), idx (row position in df) for one key kind, in parallel."""
    tasks = [(df.iloc[i:i + PART], i, kind) for i in range(0, len(df), PART)]
    with Pool(WORKERS) as pool:
        parts = pool.map(_keys_chunk, tasks)
    return pd.DataFrame({"key": np.concatenate([p[0] for p in parts]),
                         "idx": np.concatenate([p[1] for p in parts])}).drop_duplicates()


def spill_pairs(s1, other, tmp):
    """Pass 1: for each key kind, join all S1 keys with the filtered pool keys and write the
    (idx1, idx2) pairs to tmp/<kind>.parquet sorted by idx1. One kind in memory at a time."""
    tmp.mkdir(parents=True, exist_ok=True)
    for kind in KEYS:
        a, b = exploded_keys(s1, kind), exploded_keys(other, kind)
        size = b.groupby("key").idx.transform("size")
        b = b[size <= MAX_BLOCK[kind]]
        m = a.merge(b, on="key", suffixes=("1", "2"))[["idx1", "idx2"]].astype(np.int32)
        m = m.drop_duplicates().sort_values(["idx1", "idx2"])
        m.to_parquet(tmp / f"{kind}.parquet", index=False, row_group_size=1_000_000)
        print(f"  {kind}: {len(m):,} pairs", flush=True)
        del a, b, m


def chunk_pairs(tmp, lo, hi):
    """Pass 2 input: union of all kinds' pairs with lo <= idx1 < hi, one flag column per kind."""
    parts = []
    for kind in KEYS:
        m = pd.read_parquet(tmp / f"{kind}.parquet", filters=[("idx1", ">=", lo), ("idx1", "<", hi)])
        parts.append(m.assign(kind=kind))
    allp = pd.concat(parts, ignore_index=True)
    allp["kind"] = pd.Categorical(allp.kind, categories=KEYS)
    flags = pd.get_dummies(allp.kind).astype(bool)
    flags[["idx1", "idx2"]] = allp[["idx1", "idx2"]]
    return flags.groupby(["idx1", "idx2"], sort=False).max().reset_index()


def cheap_scores(pairs, s1, other):
    """Name and address skeleton similarity (0-1) for every pair, in C++ threads."""
    n1 = s1.name_skel.to_numpy()[pairs.idx1.to_numpy()]
    n2 = other.name_skel.to_numpy()[pairs.idx2.to_numpy()]
    a1 = s1.addr_skel.to_numpy()[pairs.idx1.to_numpy()]
    a2 = other.addr_skel.to_numpy()[pairs.idx2.to_numpy()]
    name = process.cpdist(n1, n2, scorer=fuzz.token_set_ratio, workers=-1) / 100.0
    addr = process.cpdist(a1, a2, scorer=fuzz.token_set_ratio, workers=-1) / 100.0
    return name.astype(np.float32), addr.astype(np.float32)


def block(split, cap=CAP):
    """Candidates for one split, capped per S1 entity and written to data/candidates/.
    For train, also returns recall statistics against the ground truth."""
    src = load_normalised(split, COLS)
    s1 = src["source1"]
    other = pd.concat([src["source2"], src["source3"]], ignore_index=True)
    tmp = DATA / "candidates" / f"_{split}_pairs"
    print(split, "keys", flush=True)
    spill_pairs(s1, other, tmp)
    truth = truth_index(s1, other) if split == "train" else None
    stats = {k: 0 for k in KEYS + ["union"] + [f"cap{c}" for c in REPORT_CAPS]}
    n_pairs = {k: 0 for k in stats}
    kept = []
    for lo in range(0, len(s1), S1_CHUNK):
        pairs = chunk_pairs(tmp, lo, lo + S1_CHUNK)
        pairs["name_sim"], pairs["addr_sim"] = cheap_scores(pairs, s1, other)
        pairs["cheap_score"] = pairs.name_sim + 0.5 * pairs.addr_sim + 0.05 * pairs[KEYS].sum(axis=1)
        rank = pairs.groupby("idx1").cheap_score.rank(method="first", ascending=False)
        if truth is not None:
            is_hit = pd.Series(pairs.merge(truth.assign(t=1), how="left", on=["idx1", "idx2"])
                               .t.notna().to_numpy(), index=pairs.index)
            for k in KEYS:
                stats[k] += int((is_hit & pairs[k]).sum())
                n_pairs[k] += int(pairs[k].sum())
            stats["union"] += int(is_hit.sum())
            n_pairs["union"] += len(pairs)
            for c in REPORT_CAPS:
                stats[f"cap{c}"] += int((is_hit & (rank <= c)).sum())
                n_pairs[f"cap{c}"] += int((rank <= c).sum())
        keep = pairs[rank <= cap]
        kept.append(pd.DataFrame({
            "s1_id": s1.entity_id.to_numpy()[keep.idx1.to_numpy()],
            "cand_id": other.entity_id.to_numpy()[keep.idx2.to_numpy()],
            **{c: keep[c].to_numpy() for c in KEYS + ["name_sim", "addr_sim", "cheap_score"]}}))
        print(f"  S1 {lo:,}+: {len(pairs):,} pairs -> {len(keep):,} kept", flush=True)
    capped = pd.concat(kept, ignore_index=True)
    capped.to_parquet(DATA / "candidates" / f"{split}_candidates.parquet", index=False)
    for f in tmp.glob("*.parquet"):
        f.unlink()
    tmp.rmdir()
    res = None
    if truth is not None:
        n_true, n_s1 = len(truth), len(s1)
        res = {**{f"recall_{k}": stats[k] / n_true for k in stats},
               **{f"cands_{k}": n_pairs[k] / n_s1 for k in n_pairs},
               "block_recall": stats[f"cap{cap}"] / n_true if cap in REPORT_CAPS else None,
               "cands_per_s1": len(capped) / n_s1}
    return capped, res


def truth_index(s1, other):
    """Ground-truth pairs as (idx1, idx2) row positions."""
    pos1 = pd.Series(np.arange(len(s1), dtype=np.int32), index=s1.entity_id.to_numpy())
    pos2 = pd.Series(np.arange(len(other), dtype=np.int32), index=other.entity_id.to_numpy())
    t = pd.DataFrame([(s, m) for s, ms in load_ground_truth().items() for m in ms], columns=["s", "m"])
    return pd.DataFrame({"idx1": pos1.reindex(t.s).to_numpy(), "idx2": pos2.reindex(t.m).to_numpy()}).dropna().astype(np.int32)


if __name__ == "__main__":
    args = sys.argv[1:]
    flags = [i for i, a in enumerate(args) if a.startswith("--")]
    splits = args[:flags[0] if flags else len(args)] or ["train", "test"]
    for split in splits:
        inputs = [DATA / "normalised" / f"{split}_{s}.parquet" for s in SOURCES]
        if "--report" not in sys.argv and is_fresh([DATA / "candidates" / f"{split}_candidates.parquet"], inputs,
                                                   ["blocking/block.py", "blocking/normalise.py"]):
            print(split, "candidates (cached)")
            continue
        capped, res = block(split)
        print(split, f"{len(capped):,} candidate pairs", flush=True)
        if res:
            for k, v in res.items():
                print(f"  {k}: {v:.4f}")
            if "--log" in sys.argv:
                i = sys.argv.index("--log")
                print(log_run(sys.argv[i + 1], sys.argv[i + 2], res))
