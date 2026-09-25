"""Candidate generation: union of blocking keys K1..K7 (PLAN.md §2.2) -> data/candidates/.

Every key requires equal country. In the train split, blocking runs separately on the fit side and
on the validation side so validation sees only its own records, exactly like test.

    python -m src.blocking.block train test [--report [--log <tag> "<change>"]]
"""
import sys
from collections import defaultdict

import jellyfish
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer

from ..common.evaluate import blocking_recall, log_run
from ..common.io_utils import DATA, SOURCES, is_fresh, pairs_to_map
from ..common.split import ground_truth_for, side_of
from ..neural import embeddings
from .normalise import STOP, load_normalised

TOPK = 20                 # K6 neighbours per S1 record, per field
MAX_BLOCK = 200           # key values with more S2/S3 records than this are too generic to use
CAP = 40                  # final candidates per S1 entity (see cap_rank)
EMB_BLOCKING = False      # K7 embedding neighbours as candidates (cosines are features either way)
CHUNK_CELLS = 2e7         # dense cells per chunk of the sparse similarity product
KEYS = ["k1", "k2", "k3", "k4", "k5", "k6", "k7"]


def key_values(df):
    """{key name: list of key strings (or None) per record} for the exact-match keys."""
    first = df["name_tokens"].str.split().str[0].fillna("")
    first_sig = [next((t for t in c.split() if t not in STOP), "") for c in df["name_core"]]
    sorted_str = df["name_tokens"].str.replace(" ", "", regex=False)
    meta = [" ".join(jellyfish.metaphone(t) for t in c.split()[:2]) for c in df["name_core"]]
    house_street = []
    for clean, nums in zip(df["addr_clean"], df["addr_numbers"]):
        toks = clean.split()
        num = nums.split()[0] if nums else ""
        word = next((t for t in toks if not t.isdigit() and len(t) > 2), "")
        house_street.append(f"{num} {word}" if num and word else None)
    return {
        "k1": [c or None for c in df["name_core"]],
        "k2": [f or None for f in first_sig],
        "k2b": [f or None for f in first],
        "k3": [s[:6] or None for s in sorted_str],
        "k4": [m or None for m in meta],
        "k5": [p or None for p in df["postal_code"]],
        "k5b": house_street,
    }


def rarest_token(df, other_df):
    """For each S1 record, its name token that is rarest among the S2/S3 records."""
    freq = defaultdict(int)
    for toks in other_df["name_tokens"]:
        for t in set(toks.split()):
            freq[t] += 1
    return [min(toks.split(), key=lambda t: freq.get(t, 0), default="") or None for toks in df["name_tokens"]]


def exact_key_pairs(s1, other):
    """Pairs sharing any exact key value; values shared by more than MAX_BLOCK records are skipped."""
    kv1, kv2 = key_values(s1), key_values(other)
    kv1["k2c"], kv2["k2c"] = rarest_token(s1, other), rarest_token(other, other)
    out = []
    for name in kv1:
        index = defaultdict(list)
        for cid, v in zip(other["entity_id"], kv2[name]):
            if v:
                index[v].append(cid)
        rows = [(sid, cid) for sid, v in zip(s1["entity_id"], kv1[name]) if v and len(index.get(v, ())) <= MAX_BLOCK
                for cid in index.get(v, ())]
        df = pd.DataFrame(rows, columns=["s1_id", "cand_id"])
        df["key"] = name[:2]
        out.append(df)
    return pd.concat(out, ignore_index=True)


def tfidf_pair(texts_a, texts_b):
    """L2-normalised char 3-gram TF-IDF matrices for two text lists, fitted on both."""
    vec = TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 3), min_df=1, sublinear_tf=True,
                          dtype=np.float32)
    vec.fit(list(texts_a) + list(texts_b))
    return vec.transform(texts_a), vec.transform(texts_b)


def topk_pairs(A, B, ids_a, ids_b, k=TOPK):
    """Top-k cosine neighbours in B for each row of A, computed in chunks of sparse products."""
    rows = max(1, int(CHUNK_CELLS // max(B.shape[0], 1)))
    k = min(k, B.shape[0])
    out_a, out_b = [], []
    BT = B.T.tocsc()
    for start in range(0, A.shape[0], rows):
        sim = (A[start:start + rows] @ BT).toarray()
        idx = np.argpartition(-sim, k - 1, axis=1)[:, :k]
        val = np.take_along_axis(sim, idx, axis=1)
        r, c = np.nonzero(val > 0)
        out_a.append(ids_a[start + r])
        out_b.append(ids_b[idx[r, c]])
    return pd.DataFrame({"s1_id": np.concatenate(out_a), "cand_id": np.concatenate(out_b)})


def dense_topk_pairs(A, B, ids_a, ids_b, k=TOPK):
    """Top-k inner-product neighbours for dense L2-normalised rows, in chunks."""
    rows = max(1, int(CHUNK_CELLS // max(B.shape[0], 1)))
    k = min(k, B.shape[0])
    out_a, out_b = [], []
    for start in range(0, A.shape[0], rows):
        sim = A[start:start + rows] @ B.T
        idx = np.argpartition(-sim, k - 1, axis=1)[:, :k]
        r = np.repeat(np.arange(sim.shape[0]), k)
        out_a.append(ids_a[start + r])
        out_b.append(ids_b[idx.ravel()])
    return pd.DataFrame({"s1_id": np.concatenate(out_a), "cand_id": np.concatenate(out_b)})


def unit_rows(M):
    """Rows scaled to unit length (zero rows stay zero)."""
    n = np.linalg.norm(M, axis=1, keepdims=True)
    return M / np.where(n > 0, n, 1.0)


def rowwise_cos(A, B, ia, ib):
    """Cosine of A[ia[i]] and B[ib[i]] for every i, both matrices L2-normalised."""
    out = np.empty(len(ia), dtype=np.float32)
    step = 200000
    for s in range(0, len(ia), step):
        out[s:s + step] = np.asarray(A[ia[s:s + step]].multiply(B[ib[s:s + step]]).sum(axis=1)).ravel()
    return out


def block_partition(s1, other, emb=None):
    """All keys for one (partition, country) group; returns pairs with key flags and cosines.
    emb: (name1, addr1, name2, addr2) embedding matrices aligned with s1 and other, or None."""
    if s1.empty or other.empty:
        return pd.DataFrame(columns=["s1_id", "cand_id", *KEYS, "ngram_name_cos", "ngram_addr_cos"])
    exact = exact_key_pairs(s1, other)
    An, Bn = tfidf_pair(s1["name_core"], other["name_core"])
    Aa, Ba = tfidf_pair(s1["addr_clean"], other["addr_clean"])
    ids1, ids2 = s1["entity_id"].to_numpy(), other["entity_id"].to_numpy()
    k6 = pd.concat([topk_pairs(An, Bn, ids1, ids2), topk_pairs(Aa, Ba, ids1, ids2)])
    k6["key"] = "k6"
    found = [exact, k6]
    if emb is not None:
        n1, a1, n2, a2 = emb
    if emb is not None and EMB_BLOCKING:
        k7 = dense_topk_pairs(unit_rows(n1 + a1), unit_rows(n2 + a2), ids1, ids2)
        k7["key"] = "k7"
        found.append(k7)
    allp = pd.concat(found, ignore_index=True)
    flags = pd.crosstab([allp.s1_id, allp.cand_id], allp.key).clip(upper=1).astype(bool)
    flags = flags.reindex(columns=KEYS, fill_value=False).reset_index()
    pos1 = pd.Series(np.arange(len(ids1)), index=ids1)
    pos2 = pd.Series(np.arange(len(ids2)), index=ids2)
    ia, ib = pos1[flags.s1_id].to_numpy(), pos2[flags.cand_id].to_numpy()
    flags["ngram_name_cos"] = rowwise_cos(An, Bn, ia, ib)
    flags["ngram_addr_cos"] = rowwise_cos(Aa, Ba, ia, ib)
    if emb is not None:
        flags["emb_name_cos"] = np.einsum("ij,ij->i", n1[ia], n2[ib])
        flags["emb_addr_cos"] = np.einsum("ij,ij->i", a1[ia], a2[ib])
        flags["emb_cos"] = np.einsum("ij,ij->i", unit_rows(n1 + a1)[ia], unit_rows(n2 + a2)[ib])
    return flags


def cap_rank(df):
    """Rank within each S1 entity by the better of the name-cosine and address-cosine ranks, ties
    broken by the cosine sum. A candidate strong on either field alone (trade names, moved
    businesses) survives the cap."""
    g = df.groupby("s1_id")
    best = np.minimum(g["ngram_name_cos"].rank(ascending=False, method="min"),
                      g["ngram_addr_cos"].rank(ascending=False, method="min"))
    score = -best + 0.001 * (df["ngram_name_cos"] + df["ngram_addr_cos"])
    return score.groupby(df["s1_id"]).rank(method="first", ascending=False)


def apply_cap(df, cap=CAP):
    """Keep the cap best candidates per S1 entity by cap_rank."""
    return df[cap_rank(df) <= cap].reset_index(drop=True)


def block(split, cap=CAP):
    """Uncapped and capped candidate frames for one split; the capped one is written."""
    src = load_normalised(split)
    s1 = src["source1"]
    other = pd.concat([src["source2"], src["source3"]], ignore_index=True)
    if split == "train":
        side = side_of(set(s1.entity_id) | set(other.entity_id))
        s1 = s1.assign(part=s1.entity_id.map(side))
        other = other.assign(part=other.entity_id.map(side))
    else:
        s1, other = s1.assign(part="all"), other.assign(part="all")
    emb = embeddings.load(split)
    row = pd.Series(np.arange(len(emb["ids"])), index=emb["ids"]) if emb is not None else None
    parts = []
    for (part, country), g1 in s1.groupby(["part", "country"], sort=True):
        g2 = other[(other.part == part) & (other.country == country)].reset_index(drop=True)
        g1 = g1.reset_index(drop=True)
        e = None
        if emb is not None:
            r1, r2 = row[g1.entity_id].to_numpy(), row[g2.entity_id].to_numpy()
            e = (emb["name"][r1], emb["addr"][r1], emb["name"][r2], emb["addr"][r2])
        parts.append(block_partition(g1, g2, e))
    full = pd.concat(parts, ignore_index=True)
    capped = apply_cap(full, cap)
    out = DATA / "candidates"
    out.mkdir(parents=True, exist_ok=True)
    capped.to_parquet(out / f"{split}_candidates.parquet", index=False)
    return full, capped


def report(full, capped):
    """Recall and candidates per S1 on the validation side, per key and for the union."""
    truth = ground_truth_for("val")
    n = len(truth)
    val_full = full[full.s1_id.isin(truth)]
    val_cap = capped[capped.s1_id.isin(truth)]
    res = {}
    for k in KEYS:
        sub = val_full[val_full[k]]
        res[f"recall_{k}"] = blocking_recall(pairs_to_map(sub), truth)
        res[f"cands_{k}"] = len(sub) / n
    for k in KEYS:
        others = val_full[val_full[[o for o in KEYS if o != k]].any(axis=1)]
        res[f"recall_without_{k}"] = blocking_recall(pairs_to_map(apply_cap(others)), truth)
    res["recall_union_uncapped"] = blocking_recall(pairs_to_map(val_full), truth)
    res["cands_union_uncapped"] = len(val_full) / n
    for cap in (10, 20, 30, 40, 60, 80):
        c = apply_cap(val_full, cap)
        res[f"recall_cap{cap}"] = blocking_recall(pairs_to_map(c), truth)
        res[f"cands_cap{cap}"] = len(c) / n
    res["block_recall"] = blocking_recall(pairs_to_map(val_cap), truth)
    res["cands_per_s1"] = len(val_cap) / n
    return res


if __name__ == "__main__":
    args = sys.argv[1:]
    flags = [i for i, a in enumerate(args) if a.startswith("--")]
    splits = args[:flags[0] if flags else len(args)] or ["train", "test"]
    for split in splits:
        inputs = [DATA / "normalised" / f"{split}_{s}.parquet" for s in SOURCES]
        if embeddings.ENABLED:
            inputs += [embeddings.OUT / f"{split}_{s}.npz" for s in SOURCES]
        if split == "train":
            inputs += [DATA / "splits" / "val_s1_ids.txt", DATA / "splits" / "val_other_ids.txt"]
        if "--report" not in sys.argv and is_fresh([DATA / "candidates" / f"{split}_candidates.parquet"], inputs):
            print(split, "candidates (cached)")
            continue
        full, capped = block(split)
        print(split, len(full), "pairs uncapped,", len(capped), "capped")
        if split == "train" and "--report" in sys.argv:
            res = report(full, capped)
            for k, v in res.items():
                print(f"  {k}: {v:.4f}")
            if "--log" in sys.argv:
                i = sys.argv.index("--log")
                print(log_run(sys.argv[i + 1], sys.argv[i + 2], res))
