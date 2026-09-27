"""Pairwise features for candidate pairs (PLAN.md §2.3) -> data/features/{split}/part_*.parquet.

All features are numeric and language-agnostic: string similarities on the normalised and
skeleton views, digit agreement, IDF-weighted token overlap, key flags from blocking, and
competition features that compare a pair with the other candidates of the same S1 entity and
with the other S1 entities that list the same S2/S3 record. Competition features come from the
whole candidate table; the string features are computed in parallel chunks and written as parts
(test has tens of millions of pairs). In train, only a fixed sample of S1 entities is featurised
(SAMPLE_FIT from the fit side, SAMPLE_VAL from the validation side).

    python -m src.matching.features train test
"""
import os
import re
import sys
from collections import Counter
from multiprocessing import Pool

import numpy as np
import pandas as pd
import psutil
import pyarrow as pa
import pyarrow.compute as pc
from rapidfuzz import fuzz, process
from rapidfuzz.distance import JaroWinkler

from ..blocking.block import KEYS
from ..blocking.normalise import load_normalised
from ..common.io_utils import DATA, RAW, SOURCES, is_fresh, load_ground_truth
from ..common.split import load_split
from .ranking import second_largest

SAMPLE_FIT, SAMPLE_VAL = 200_000, 60_000
PART_S1 = 25_000                  # S1 entities per output part (keeps Python strings per part small)
CHUNK = 100_000                   # pairs per worker task
WORKERS = 6                       # each worker holds its task's strings; 6 leaves RAM headroom
LANDMARK = {"nr", "opp", "bsd", "bhd"}
TEXT = ["business_name", "name_clean", "name_core", "legal_suffix", "name_tokens", "name_skel", "name_alt",
        "addr_clean", "addr_numbers", "postal_code", "addr_tokens", "city_guess", "addr_skel"]
_IDF = {}
# candidate-side texts kept in the feature parts (not model features; used by train_lgbm stage 2)
SIBLING_TEXT = {"sib_name": "name_skel", "sib_addr": "addr_skel", "sib_num": "addr_numbers",
                "sib_raw": "business_name"}
CODE = ["matching/features.py", "matching/ranking.py", "blocking/block.py", "common/split.py"]


def sim(a, b, scorer):
    """Element-wise rapidfuzz scores in [0, 1] for two aligned string arrays."""
    scale = 1.0 if scorer is JaroWinkler.normalized_similarity else 100.0
    return (process.cpdist(a, b, scorer=scorer, workers=1) / scale).astype(np.float32)


def jaccard(a, b):
    """Token-set Jaccard for aligned lists of space-separated token strings (0 if both empty)."""
    out = np.zeros(len(a), dtype=np.float32)
    for i, (x, y) in enumerate(zip(a, b)):
        sx, sy = set(x.split()), set(y.split())
        if sx or sy:
            out[i] = len(sx & sy) / len(sx | sy)
    return out


def idf_table(texts):
    """{token: idf} over an iterable of token strings. Tokens seen once all share the largest idf,
    so they are left out and that value is stored under the key " " (no token contains a space);
    lookups of absent tokens fall back to it. Keeps the table small enough to copy to workers."""
    df, n = Counter(), 0
    for t in texts:
        df.update(set(t.split()))
        n += 1
    table = {k: float(np.log((1 + n) / (1 + v))) + 1.0 for k, v in df.items() if v > 1}
    table[" "] = float(np.log((1 + n) / 2)) + 1.0
    return table


def idf_overlap(a, b, idf):
    """IDF-weighted Jaccard, IDF cosine and the IDF of the rarest shared token."""
    wj = np.zeros(len(a), dtype=np.float32)
    cos = np.zeros(len(a), dtype=np.float32)
    rare = np.zeros(len(a), dtype=np.float32)
    top = idf.get(" ", 1.0)
    for i, (x, y) in enumerate(zip(a, b)):
        sx, sy = set(x.split()), set(y.split())
        w = lambda t: idf.get(t, top)
        shared = [w(t) for t in sx & sy]
        union = sum(w(t) for t in sx | sy)
        if union:
            wj[i] = sum(shared) / union
            nx, ny = sum(w(t) ** 2 for t in sx) ** 0.5, sum(w(t) ** 2 for t in sy) ** 0.5
            cos[i] = sum(s * s for s in shared) / (nx * ny) if nx and ny else 0.0
        rare[i] = max(shared, default=0.0)
    return wj, cos, rare


def common_prefix(a, b):
    out = np.zeros(len(a), dtype=np.float32)
    for i, (x, y) in enumerate(zip(a, b)):
        n = 0
        for cx, cy in zip(x, y):
            if cx != cy:
                break
            n += 1
        out[i] = n
    return out


def acronym_match(a, b):
    """1 if a token of one side equals (or starts) the initials of the other side's tokens."""
    out = np.zeros(len(a), dtype=np.float32)
    for i, (x, y) in enumerate(zip(a, b)):
        tx, ty = x.split(), y.split()
        ix, iy = "".join(t[0] for t in tx), "".join(t[0] for t in ty)
        if any(len(t) > 1 and iy.startswith(t) for t in tx) or \
           any(len(t) > 1 and ix.startswith(t) for t in ty):
            out[i] = 1.0
    return out


def tri_state(a, b):
    """0 both empty, 1 equal, 2 one empty, 3 conflict."""
    out = np.empty(len(a), dtype=np.float32)
    for i, (x, y) in enumerate(zip(a, b)):
        out[i] = 0 if not x and not y else 1 if x == y else 2 if not x or not y else 3
    return out


def token_in(a_tokens, b_text):
    """Share of tokens of a (len > 2) that appear in the token set of b."""
    out = np.zeros(len(a_tokens), dtype=np.float32)
    for i, (x, y) in enumerate(zip(a_tokens, b_text)):
        tx = [t for t in x.split() if len(t) > 2]
        if tx:
            sy = set(y.split())
            out[i] = sum(t in sy for t in tx) / len(tx)
    return out


NON_LATIN = re.compile(r"[^\W\d_A-Za-zÀ-ɏ]")
LETTER = re.compile(r"[^\W\d_]")


def script_share(names):
    """Share of letters outside the Latin script (Devanagari, Tamil, ...) in each raw name: a
    transliterated name is expected to differ from its Latin spelling more than a typo does."""
    out = np.zeros(len(names), dtype=np.float32)
    for i, n in enumerate(names):
        letters = len(LETTER.findall(n))
        if letters:
            out[i] = len(NON_LATIN.findall(n)) / letters
    return out


# legal forms grouped by kind: sources rewrite "Private Limited" as "Pvt Ltd" or drop words, but a
# different kind (LLP vs Ltd, PLLC vs Co) marks a different business
SUFFIX_FAMILY = {"pvt": "ltd", "ltd": "ltd", "plc": "ltd", "pty": "ltd", "opc": "ltd", "llp": "llp",
                 "lp": "llp", "llc": "llc", "inc": "corp", "corp": "corp", "co": "co", "pc": "pc",
                 "pa": "pc", "sarl": "sarl", "sas": "sas", "sa": "sa", "eurl": "eurl", "sci": "sci",
                 "snc": "snc", "gmbh": "gmbh", "ag": "ag", "bv": "bv", "nv": "nv", "srl": "srl",
                 "spa": "spa"}


def suffix_family_conflict(a, b):
    """1 when both names carry a legal form and the kinds are disjoint."""
    out = np.zeros(len(a), dtype=np.float32)
    for i, (x, y) in enumerate(zip(a, b)):
        fx = {SUFFIX_FAMILY.get(t, t) for t in x.split()}
        fy = {SUFFIX_FAMILY.get(t, t) for t in y.split()}
        if fx and fy and not fx & fy:
            out[i] = 1.0
    return out


def number_substitution(a, b):
    """(share of the shorter side's numbers found exactly on the other side, 1 if some number was
    substituted rather than dropped: after removing exact counterparts, a leftover number on one
    side has a same-length leftover on the other side differing in one or two digits). Dropped or
    added digits are source noise; a changed digit marks a neighbouring unit or building."""
    exact = np.zeros(len(a), dtype=np.float32)
    sub = np.zeros(len(a), dtype=np.float32)
    for i, (x, y) in enumerate(zip(a, b)):
        nx, ny = x.split(), y.split()
        if not nx or not ny:
            continue
        cx, cy = Counter(nx), Counter(ny)
        exact[i] = sum((cx & cy).values()) / min(len(nx), len(ny))
        lx, ly = cx - cy, cy - cx
        if any(len(u) == len(v) and 0 < sum(c != d for c, d in zip(u, v)) <= 2 for u in lx for v in ly):
            sub[i] = 1.0
    return exact, sub


# words the sources append to names (seen in matched groups): not evidence of a different business
FILLER = {"services", "service", "center", "centre", "partners", "group", "mr", "shri", "sri", "smt"}


def word_swap(a, b):
    """(number of words on the S1 side replaced by a dissimilar word on the other side, number of
    words only the other side has). Sources add filler words or misspell words; replacing one word
    by a different one ("Gdb Logistics" / "Gdb Power") marks a different business."""
    swap = np.zeros(len(a), dtype=np.float32)
    extra = np.zeros(len(a), dtype=np.float32)
    for i, (x, y) in enumerate(zip(a, b)):
        sx, sy = set(x.split()) - FILLER, set(y.split()) - FILLER
        ux, uy = sx - sy, sy - sx
        if ux and uy:
            swap[i] = sum(max(fuzz.ratio(u, v) for v in uy) < 60 for u in ux)
        extra[i] = len(uy) if not ux else 0
    return swap, extra


def tokens_in_glued(names, glued):
    """Share of a name's words (3+ letters) found inside the other name written without spaces:
    "dass priya" is fully inside "priyadass" (domain-style and concatenated names)."""
    out = np.zeros(len(names), dtype=np.float32)
    for i, (x, g) in enumerate(zip(names, glued)):
        words = [w for w in x.split() if len(w) >= 3]
        if words and g:
            out[i] = sum(w in g for w in words) / len(words)
    return out


def number_contained(a, b):
    """Share of the shorter side's address numbers that appear in, or are a digit-dropped part of,
    a number on the other side (house numbers lose or gain a digit between sources)."""
    out = np.zeros(len(a), dtype=np.float32)
    for i, (x, y) in enumerate(zip(a, b)):
        nx, ny = x.split(), y.split()
        if not nx or not ny:
            continue
        short, long_ = (nx, ny) if len(nx) <= len(ny) else (ny, nx)
        hit = sum(any(u == v or (len(u) >= 2 and (u in v or v in u)) for v in long_) for u in short)
        out[i] = hit / len(short)
    return out


def length_ratio(a, b):
    la = np.fromiter((len(x) for x in a), dtype=np.float32, count=len(a))
    lb = np.fromiter((len(x) for x in b), dtype=np.float32, count=len(b))
    return np.minimum(la, lb) / np.maximum(np.maximum(la, lb), 1)


def string_features(A, B):
    """String features for aligned record frames A (S1 side) and B (S2/S3 side)."""
    f = {}
    nc1, nc2 = A.name_core.to_numpy(), B.name_core.to_numpy()
    f["name_jw"] = sim(nc1, nc2, JaroWinkler.normalized_similarity)
    f["name_ratio"] = sim(nc1, nc2, fuzz.ratio)
    f["name_token_set"] = sim(nc1, nc2, fuzz.token_set_ratio)
    f["name_token_sort"] = sim(nc1, nc2, fuzz.token_sort_ratio)
    f["name_partial"] = sim(nc1, nc2, fuzz.partial_ratio)
    f["name_clean_ratio"] = sim(A.name_clean.to_numpy(), B.name_clean.to_numpy(), fuzz.ratio)
    s1k, s2k = A.name_skel.to_numpy(), B.name_skel.to_numpy()
    f["skel_ratio"] = sim(s1k, s2k, fuzz.ratio)
    f["skel_token_set"] = sim(s1k, s2k, fuzz.token_set_ratio)
    f["skel_partial"] = sim(s1k, s2k, fuzz.partial_ratio)
    f["skel_jaccard"] = jaccard(s1k, s2k)
    alt2 = np.where(B.name_alt.to_numpy() != "", B.name_alt.to_numpy(), nc2)
    f["name_alt_token_set"] = np.maximum(sim(nc1, alt2, fuzz.token_set_ratio), f["name_token_set"])
    f["name_jaccard"] = jaccard(A.name_tokens, B.name_tokens)
    f["name_idf_jaccard"], f["name_idf_cos"], f["name_rare_shared_idf"] = idf_overlap(
        A.name_skel, B.name_skel, _IDF["name"])
    f["name_prefix_len"] = common_prefix(nc1, nc2)
    f["name_first_tok_eq"] = (A.name_tokens.str.split().str[0].fillna("").to_numpy() ==
                              B.name_tokens.str.split().str[0].fillna("").to_numpy()).astype(np.float32)
    f["name_acronym"] = acronym_match(nc1, nc2)
    f["suffix_state"] = tri_state(A.legal_suffix, B.legal_suffix)
    f["name_ntok_1"] = A.name_tokens.str.count(" ").to_numpy(dtype=np.float32) + (A.name_tokens != "").to_numpy()
    f["name_ntok_2"] = B.name_tokens.str.count(" ").to_numpy(dtype=np.float32) + (B.name_tokens != "").to_numpy()
    f["name_len_ratio"] = length_ratio(nc1, nc2)
    f["script_1"] = script_share(A.business_name.to_numpy())
    f["script_2"] = script_share(B.business_name.to_numpy())
    f["script_diff"] = np.abs(f["script_1"] - f["script_2"])

    ac1, ac2 = A.addr_clean.to_numpy(), B.addr_clean.to_numpy()
    f["addr_ratio"] = sim(ac1, ac2, fuzz.ratio)
    f["addr_token_set"] = sim(ac1, ac2, fuzz.token_set_ratio)
    f["addr_token_sort"] = sim(ac1, ac2, fuzz.token_sort_ratio)
    f["addr_partial"] = sim(ac1, ac2, fuzz.partial_ratio)
    f["addr_skel_token_set"] = sim(A.addr_skel.to_numpy(), B.addr_skel.to_numpy(), fuzz.token_set_ratio)
    f["addr_jaccard"] = jaccard(A.addr_tokens, B.addr_tokens)
    f["addr_idf_jaccard"], f["addr_idf_cos"], f["addr_rare_shared_idf"] = idf_overlap(
        A.addr_skel, B.addr_skel, _IDF["addr"])
    f["addr_num_jaccard"] = jaccard(A.addr_numbers, B.addr_numbers)
    h1 = A.addr_numbers.str.split().str[0].fillna("")
    h2 = B.addr_numbers.str.split().str[0].fillna("")
    f["house_state"] = tri_state(h1, h2)
    f["house_in_other"] = token_in(h1.to_numpy(), B.addr_numbers.to_numpy())
    f["postal_state"] = tri_state(A.postal_code, B.postal_code)
    f["city_jaccard"] = jaccard(A.city_guess, B.city_guess)
    f["addr_len_ratio"] = length_ratio(ac1, ac2)
    f["landmark_any"] = np.fromiter((bool(LANDMARK & (set(x.split()) | set(y.split())))
                                     for x, y in zip(ac1, ac2)), dtype=np.float32, count=len(ac1))
    f["addr_empty_1"] = (A.addr_clean == "").to_numpy(dtype=np.float32)
    f["addr_empty_2"] = (B.addr_clean == "").to_numpy(dtype=np.float32)
    f["name1_in_addr2"] = token_in(A.name_tokens, B.addr_clean)
    f["house_ratio"] = sim(h1.to_numpy(), h2.to_numpy(), fuzz.ratio)          # 12573 vs 2573: 0.89
    f["num_contained"] = number_contained(A.addr_numbers.to_numpy(), B.addr_numbers.to_numpy())
    ns1 = np.array([x.replace(" ", "") for x in nc1], dtype=object)
    ns2 = np.array([x.replace(" ", "") for x in nc2], dtype=object)
    f["name_nospace_ratio"] = sim(ns1, ns2, fuzz.ratio)                      # "priya dass" vs "priyadass"
    f["name_nospace_partial"] = sim(ns1, ns2, fuzz.partial_ratio)
    f["name_glued"] = np.maximum(tokens_in_glued(nc1, ns2), tokens_in_glued(nc2, ns1))
    f["suffix_family_conflict"] = suffix_family_conflict(A.legal_suffix.to_numpy(), B.legal_suffix.to_numpy())
    f["num_exact_share"], f["num_substituted"] = number_substitution(A.addr_numbers.to_numpy(),
                                                                     B.addr_numbers.to_numpy())
    f["name_word_swap"], f["name_extra_words"] = word_swap(A.name_tokens.to_numpy(), B.name_tokens.to_numpy())
    f["name2_in_addr1"] = token_in(B.name_tokens, A.addr_clean)
    return pd.DataFrame(f)


def _init(idf):
    _IDF.update(idf)


def _chunk(args):
    A, B = args
    return string_features(A.reset_index(drop=True), B.reset_index(drop=True))


def competition(df, cols):
    """Context features over a candidate table: rank, gap and margin of a pair among the candidates
    of its S1 entity, and the best score its S2/S3 record reaches with any other S1 entity.
    Groups by integer codes, so tens of millions of rows fit in memory."""
    k1 = pd.factorize(df.s1_id)[0]
    k2 = pd.factorize(df.cand_id)[0]
    g1, g2 = df.groupby(k1, sort=False), df.groupby(k2, sort=False)
    out = pd.DataFrame(index=df.index)
    out["n_cands_s1"] = g1[cols[0]].transform("size").astype(np.float32)
    out["n_s1_for_cand"] = g2[cols[0]].transform("size").astype(np.float32)
    for c in cols:
        v = df[c].to_numpy()
        best = g1[c].transform("max").to_numpy()
        out[f"{c}_rank_s1"] = g1[c].rank(ascending=False, method="min").astype(np.float32)
        out[f"{c}_gap_s1"] = (best - v).astype(np.float32)
        second = second_largest(v, k1)
        out[f"{c}_margin_s1"] = np.where(v >= best, v - second, v - best).astype(np.float32)
        top1 = g2[c].transform("max").to_numpy()
        other_best = np.where(v >= top1, second_largest(v, k2), top1)
        out[f"{c}_other_s1_best"] = other_best.astype(np.float32)
        out[f"{c}_margin_cand"] = (v - other_best).astype(np.float32)
        out[f"{c}_rank_cand"] = g2[c].rank(ascending=False, method="min").astype(np.float32)
    return out


def train_sample():
    """{s1_id: 'fit' | 'val'} for the fixed training sample (seed 42)."""
    val_s1 = load_split()
    s1 = load_normalised("train", ["entity_id"])["source1"].entity_id.to_numpy()
    rng = np.random.RandomState(42)
    is_val = pd.Series(s1).isin(val_s1).to_numpy()   # hash lookup; np.isin on objects is quadratic
    fit = rng.choice(s1[~is_val], min(SAMPLE_FIT, (~is_val).sum()), replace=False)
    val = rng.choice(s1[is_val], min(SAMPLE_VAL, is_val.sum()), replace=False)
    return {**{s: "fit" for s in fit}, **{s: "val" for s in val}}


def frequency_tables(s1, other):
    """How ambiguous a record's name and address are within its country, as log counts per record
    id: how many S1 entities share the S1 record's name skeleton / address key, and how many share
    the candidate's. A record without an address whose name no other S1 entity has is almost
    surely a match; an identical address shared by one S1 entity carries a renamed record."""
    def key_addr(df):
        first = df.addr_numbers.str.split(" ").str[0].fillna("")
        return first + "|" + df.addr_skel.astype(str)
    s1_name, s1_addr = s1.name_skel.astype(str), key_addr(s1)
    ot_name, ot_addr = other.name_skel.astype(str), key_addr(other)
    n_name, n_addr = s1_name.value_counts(), s1_addr.value_counts()
    o_name = ot_name.value_counts()
    log = lambda x: np.log1p(x.fillna(0).astype(np.float64))
    empty_addr = lambda k: k.str.endswith("|")          # no numbers and no address words
    return {
        "s1_name_freq": (log(s1_name.map(n_name)), "s1_id"),
        "s1_addr_freq": (log(s1_addr.map(n_addr)).where(~empty_addr(s1_addr)), "s1_id"),
        "cand_name_freq_s1": (log(ot_name.map(n_name)), "cand_id"),
        "cand_name_freq_other": (log(ot_name.map(o_name)), "cand_id"),
        "cand_addr_freq_s1": (log(ot_addr.map(n_addr)).where(~empty_addr(ot_addr)), "cand_id"),
    }


def load_country(split, country):
    """Normalised text views of one country: (S1 frame, S2/S3 frame), indexed by entity_id."""
    cols = ["entity_id"] + TEXT
    read = lambda src: pd.read_parquet(DATA / "normalised" / f"{split}_{src}.parquet", columns=cols,
                                       filters=[("country", "==", country)], dtype_backend="pyarrow")
    s1 = read("source1").set_index("entity_id")
    other = pd.concat([read("source2"), read("source3")]).set_index("entity_id")
    return s1, other


def build(split):
    """Features for one split, one country at a time (keys never cross countries), written as
    parts; returns the number of rows."""
    cands = pd.read_parquet(DATA / "candidates" / f"{split}_candidates.parquet", dtype_backend="pyarrow")
    for k in KEYS:
        cands[k] = cands[k].astype(np.float32)
    for c in ("name_sim", "addr_sim", "cheap_score"):
        cands[c] = cands[c].astype(np.float32)
    s1c = pd.read_parquet(DATA / "normalised" / f"{split}_source1.parquet", columns=["entity_id", "country"])
    names, codes = np.unique(s1c.country.to_numpy(dtype=object), return_inverse=True)
    pos = pc.index_in(pa.array(cands.s1_id.array), value_set=pa.array(s1c.entity_id.to_numpy(dtype=object)))
    ccode = codes.astype(np.int16)[pos.to_numpy(zero_copy_only=False)]   # C++ lookup, no Python strings
    side = train_sample() if split == "train" else None
    truth = None
    if side is not None:
        truth = pd.DataFrame([(s, m) for s, ms in load_ground_truth().items() if s in side for m in ms],
                             columns=["s1_id", "cand_id"]).assign(label=1)
    out = DATA / "features" / split
    out.mkdir(parents=True, exist_ok=True)
    resume = os.environ.get("RESUME") == "1"   # keep parts written by an interrupted run
    if not resume:
        for old in out.glob("part_*.parquet"):
            old.unlink()
    rss = lambda: psutil.Process().memory_info().rss / 1e9
    total = 0
    for ci, cname in enumerate(names):
        sub = cands[ccode == ci]
        if side is not None:                  # every row of every candidate a sampled S1 lists
            listed = sub.cand_id[sub.s1_id.isin(side.keys())].unique()
            sub = sub[sub.cand_id.isin(listed)]
        sub = sub.reset_index(drop=True)
        base = pd.concat([sub, competition(sub, ["cheap_score", "name_sim"])], axis=1)
        del sub
        if side is not None:
            base = base[base.s1_id.isin(side.keys())].reset_index(drop=True)
            base["side"] = base.s1_id.map(side)
            base = base.merge(truth, on=["s1_id", "cand_id"], how="left")
            base["label"] = base.label.fillna(0).astype(np.int8)
        s1, other = load_country(split, cname)
        idf = {"name": idf_table(pd.concat([s1.name_skel, other.name_skel])),
               "addr": idf_table(pd.concat([s1.addr_skel, other.addr_skel]))}
        freq = frequency_tables(s1, other)
        ids = base.s1_id.unique()
        with Pool(WORKERS, initializer=_init, initargs=(idf,)) as pool:
            for p, start in enumerate(range(0, len(ids), PART_S1)):
                path = out / f"part_{ci}{p:03d}.parquet"
                if resume and path.exists():
                    continue
                part = base[base.s1_id.isin(ids[start:start + PART_S1])].reset_index(drop=True)
                A, B = s1.loc[part.s1_id].astype(object), other.loc[part.cand_id].astype(object)
                tasks = [(A.iloc[i:i + CHUNK], B.iloc[i:i + CHUNK]) for i in range(0, len(part), CHUNK)]
                feats = pd.concat(pool.map(_chunk, tasks), ignore_index=True)
                for col, (table, ids) in freq.items():
                    feats[col] = table.reindex(part[ids].to_numpy()).to_numpy(np.float32)
                for col, src in SIBLING_TEXT.items():   # candidate texts for stage-2 sibling features
                    feats[col] = B[src].to_numpy()
                pd.concat([part, feats], axis=1).to_parquet(path, index=False)
                print(f"  {split} {cname} part {p}: {len(part):,} pairs, rss {rss():.1f} GB", flush=True)
        total += len(base)
        del base, s1, other
    return total


def load_features(split, columns=None):
    """All feature parts of a split as one frame (train sample; use parts() for test)."""
    return pd.concat([pd.read_parquet(f, columns=columns) for f in parts(split)], ignore_index=True)


def parts(split):
    return sorted((DATA / "features" / split).glob("part_*.parquet"))


def inputs_of(split):
    """Files the feature parts of a split are derived from."""
    files = [DATA / "candidates" / f"{split}_candidates.parquet"]
    files += [DATA / "normalised" / f"{split}_{s}.parquet" for s in SOURCES]
    if split == "train":
        files += [RAW / "train" / "train_ground_truth.tsv", DATA / "splits" / "val_s1_ids.txt"]
    return files


if __name__ == "__main__":
    for split in sys.argv[1:] or ("train", "test"):
        done = parts(split)
        if done and is_fresh(done, inputs_of(split), CODE):
            print(split, "features (cached)")
            continue
        print(split, f"{build(split):,} rows", flush=True)
