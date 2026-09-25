"""Pairwise features for every candidate pair (PLAN.md §2.3) -> data/features/{split}_features.parquet.

All features are numeric and language-agnostic: string similarities on the normalised views, digit
agreement, IDF-weighted token overlap, key flags from blocking, and competition features that
compare a pair with the other candidates of the same S1 entity and of the same S2/S3 record.

    python -m src.matching.features train test
"""
import sys

import numpy as np
import pandas as pd
from rapidfuzz import fuzz, process
from rapidfuzz.distance import JaroWinkler
from sklearn.feature_extraction.text import TfidfVectorizer

from ..blocking.block import KEYS, rowwise_cos
from ..blocking.normalise import load_normalised
from ..common.io_utils import DATA, RAW, SOURCES, is_fresh, load_ground_truth
from ..common.split import side_of

LANDMARK = {"nr", "opp", "bsd", "bhd"}


def sim(a, b, scorer):
    """Element-wise rapidfuzz scores in [0, 1] for two aligned string lists, all cores."""
    return process.cpdist(a, b, scorer=scorer, workers=-1).astype(np.float32) / (
        1.0 if scorer is JaroWinkler.normalized_similarity else 100.0)


def jaccard(a, b):
    """Token-set Jaccard for aligned lists of space-separated token strings (0 if both empty)."""
    out = np.zeros(len(a), dtype=np.float32)
    for i, (x, y) in enumerate(zip(a, b)):
        sx, sy = set(x.split()), set(y.split())
        if sx or sy:
            out[i] = len(sx & sy) / len(sx | sy)
    return out


def idf_table(texts):
    """{token: idf} over a list of token strings."""
    df = {}
    for t in texts:
        for tok in set(t.split()):
            df[tok] = df.get(tok, 0) + 1
    n = len(texts)
    return {k: float(np.log((1 + n) / (1 + v))) + 1.0 for k, v in df.items()}


def idf_overlap(a, b, idf):
    """IDF-weighted Jaccard and the IDF of the rarest shared token."""
    wj = np.zeros(len(a), dtype=np.float32)
    rare = np.zeros(len(a), dtype=np.float32)
    for i, (x, y) in enumerate(zip(a, b)):
        sx, sy = set(x.split()), set(y.split())
        union = sum(idf.get(t, 0.0) for t in sx | sy)
        shared = [idf.get(t, 0.0) for t in sx & sy]
        if union:
            wj[i] = sum(shared) / union
        rare[i] = max(shared, default=0.0)
    return wj, rare


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
    """1 if a single-token side equals the initials of the other side's tokens."""
    out = np.zeros(len(a), dtype=np.float32)
    for i, (x, y) in enumerate(zip(a, b)):
        tx, ty = x.split(), y.split()
        ix, iy = "".join(t[0] for t in tx), "".join(t[0] for t in ty)
        if any(len(t) > 1 and (t == iy or iy.startswith(t)) for t in tx) or \
           any(len(t) > 1 and (t == ix or ix.startswith(t)) for t in ty):
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


def text_cos(texts1, texts2, i1, i2, **tfidf_args):
    """Row-wise TF-IDF cosine for pairs given index arrays into two text lists."""
    vec = TfidfVectorizer(dtype=np.float32, sublinear_tf=True, **tfidf_args)
    vec.fit(list(texts1) + list(texts2))
    return rowwise_cos(vec.transform(texts1), vec.transform(texts2), i1, i2)


def second_largest(values, groups):
    """Per-row second-largest value of its group (0 for groups of one), without Python loops."""
    v = np.asarray(values, dtype=np.float64)
    codes = pd.factorize(np.asarray(groups))[0]
    order = np.lexsort((-v, codes))
    sc, sv = codes[order], v[order]
    start = np.r_[True, sc[1:] != sc[:-1]]
    first_pos = np.flatnonzero(start)
    size = np.diff(np.r_[first_pos, len(sv)])
    second = np.where(size > 1, sv[np.minimum(first_pos + 1, len(sv) - 1)], 0.0)
    return second[codes]


def competition(df, cols):
    """Context features: how a pair compares with the other candidates of its S1 entity and with
    the other S1 entities that list the same S2/S3 record. `cols` are score columns to use."""
    g1, g2 = df.groupby("s1_id"), df.groupby("cand_id")
    df["n_cands_s1"] = g1["cand_id"].transform("size").astype(np.float32)
    df["n_s1_for_cand"] = g2["s1_id"].transform("size").astype(np.float32)
    for c in cols:
        best = g1[c].transform("max")
        df[f"{c}_rank_s1"] = g1[c].rank(ascending=False, method="min").astype(np.float32)
        df[f"{c}_gap_s1"] = (best - df[c]).astype(np.float32)
        second = second_largest(df[c], df.s1_id)
        df[f"{c}_margin_s1"] = np.where(df[c] >= best, df[c] - second, df[c] - best).astype(np.float32)
        top1 = g2[c].transform("max")
        top2 = second_largest(df[c], df.cand_id)
        other_best = np.where(df[c] >= top1, top2, top1)
        df[f"{c}_other_s1_best"] = other_best.astype(np.float32)
        df[f"{c}_margin_cand"] = (df[c] - other_best).astype(np.float32)
        df[f"{c}_rank_cand"] = g2[c].rank(ascending=False, method="min").astype(np.float32)
    return df


def build(split):
    src = load_normalised(split)
    s1 = src["source1"].reset_index(drop=True)
    other = pd.concat([src["source2"], src["source3"]], ignore_index=True)
    cands = pd.read_parquet(DATA / "candidates" / f"{split}_candidates.parquet")
    pos1 = pd.Series(np.arange(len(s1)), index=s1.entity_id)
    pos2 = pd.Series(np.arange(len(other)), index=other.entity_id)
    i1, i2 = pos1[cands.s1_id].to_numpy(), pos2[cands.cand_id].to_numpy()
    A, B = s1.iloc[i1].reset_index(drop=True), other.iloc[i2].reset_index(drop=True)

    df = cands[["s1_id", "cand_id"]].copy()
    for k in KEYS:
        df[k] = cands[k].astype(np.float32).to_numpy()
    df["name_char_cos"] = cands["ngram_name_cos"].to_numpy()
    df["addr_char_cos"] = cands["ngram_addr_cos"].to_numpy()
    for c in ("emb_name_cos", "emb_addr_cos", "emb_cos"):
        if c in cands:
            df[c] = cands[c].to_numpy()

    nc1, nc2 = A.name_core.tolist(), B.name_core.tolist()
    df["name_jw"] = sim(nc1, nc2, JaroWinkler.normalized_similarity)
    df["name_ratio"] = sim(nc1, nc2, fuzz.ratio)
    df["name_token_set"] = sim(nc1, nc2, fuzz.token_set_ratio)
    df["name_token_sort"] = sim(nc1, nc2, fuzz.token_sort_ratio)
    df["name_partial"] = sim(nc1, nc2, fuzz.partial_ratio)
    df["name_clean_ratio"] = sim(A.name_clean.tolist(), B.name_clean.tolist(), fuzz.ratio)
    df["name_jaccard"] = jaccard(A.name_tokens, B.name_tokens)
    all_names = pd.concat([s1.name_tokens, other.name_tokens]).tolist()
    idf = idf_table(all_names)
    df["name_idf_jaccard"], df["name_rare_shared_idf"] = idf_overlap(A.name_tokens, B.name_tokens, idf)
    df["name_word_cos"] = text_cos(s1.name_tokens, other.name_tokens, i1, i2, analyzer="word",
                                   token_pattern=r"\S+")
    df["name_prefix_len"] = common_prefix(nc1, nc2)
    df["name_first_tok_eq"] = (A.name_tokens.str.split().str[0] == B.name_tokens.str.split().str[0]
                               ).astype(np.float32).to_numpy()
    df["name_acronym"] = acronym_match(nc1, nc2)
    df["suffix_state"] = tri_state(A.legal_suffix, B.legal_suffix)
    df["name_ntok_1"] = A.name_tokens.str.split().str.len().astype(np.float32).to_numpy()
    df["name_ntok_2"] = B.name_tokens.str.split().str.len().astype(np.float32).to_numpy()
    df["name_len_ratio"] = (np.minimum(A.name_core.str.len(), B.name_core.str.len()) /
                            np.maximum(A.name_core.str.len(), B.name_core.str.len()).clip(lower=1)
                            ).astype(np.float32).to_numpy()

    ac1, ac2 = A.addr_clean.tolist(), B.addr_clean.tolist()
    df["addr_ratio"] = sim(ac1, ac2, fuzz.ratio)
    df["addr_token_set"] = sim(ac1, ac2, fuzz.token_set_ratio)
    df["addr_token_sort"] = sim(ac1, ac2, fuzz.token_sort_ratio)
    df["addr_partial"] = sim(ac1, ac2, fuzz.partial_ratio)
    df["addr_jaccard"] = jaccard(A.addr_tokens, B.addr_tokens)
    aidf = idf_table(pd.concat([s1.addr_tokens, other.addr_tokens]).tolist())
    df["addr_idf_jaccard"], df["addr_rare_shared_idf"] = idf_overlap(A.addr_tokens, B.addr_tokens, aidf)
    df["addr_num_jaccard"] = jaccard(A.addr_numbers, B.addr_numbers)
    h1, h2 = A.addr_numbers.str.split().str[0].fillna(""), B.addr_numbers.str.split().str[0].fillna("")
    df["house_state"] = tri_state(h1, h2)
    df["postal_state"] = tri_state(A.postal_code, B.postal_code)
    df["postal_prefix3_eq"] = ((A.postal_code.str[:3] == B.postal_code.str[:3]) &
                               (A.postal_code != "")).astype(np.float32).to_numpy()
    df["city_jaccard"] = jaccard(A.city_guess, B.city_guess)
    df["addr_len_ratio"] = (np.minimum(A.addr_clean.str.len(), B.addr_clean.str.len()) /
                            np.maximum(A.addr_clean.str.len(), B.addr_clean.str.len()).clip(lower=1)
                            ).astype(np.float32).to_numpy()
    df["landmark_any"] = [float(bool(LANDMARK & (set(x.split()) | set(y.split())))) for x, y in zip(ac1, ac2)]
    df["addr_empty_any"] = ((A.addr_clean == "") | (B.addr_clean == "")).astype(np.float32).to_numpy()

    df["name1_in_addr2"] = token_in(A.name_tokens, B.addr_clean)
    df["name2_in_addr1"] = token_in(B.name_tokens, A.addr_clean)
    both1 = (s1.name_core + " " + s1.addr_clean).tolist()
    both2 = (other.name_core + " " + other.addr_clean).tolist()
    df["both_char_cos"] = text_cos(both1, both2, i1, i2, analyzer="char_wb", ngram_range=(3, 3))
    df["is_s3"] = cands.cand_id.str.startswith("S3-").astype(np.float32).to_numpy()

    df["pair_score"] = (df.name_char_cos + df.addr_char_cos).astype(np.float32)
    df = competition(df, ["name_char_cos", "addr_char_cos", "pair_score", "name_token_set"]
                     + (["emb_cos"] if "emb_cos" in df else []))

    if split == "train":
        df["side"] = df.s1_id.map(side_of(set(df.s1_id)))
        truth = {(s, m) for s, ms in load_ground_truth().items() for m in ms}
        df["label"] = [int((s, c) in truth) for s, c in zip(df.s1_id, df.cand_id)]
    out = DATA / "features"
    out.mkdir(parents=True, exist_ok=True)
    df.to_parquet(out / f"{split}_features.parquet", index=False)
    return df


def inputs_of(split):
    """Files the feature table of a split is derived from."""
    files = [DATA / "candidates" / f"{split}_candidates.parquet"]
    files += [DATA / "normalised" / f"{split}_{s}.parquet" for s in SOURCES]
    if split == "train":
        files += [RAW / "train" / "train_ground_truth.tsv", DATA / "splits" / "val_s1_ids.txt"]
    return files


if __name__ == "__main__":
    for split in sys.argv[1:] or ("train", "test"):
        if is_fresh([DATA / "features" / f"{split}_features.parquet"], inputs_of(split)):
            print(split, "features (cached)")
            continue
        print(split, build(split).shape)
