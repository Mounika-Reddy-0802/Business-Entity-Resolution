"""Pairwise features for every candidate pair -> data/features/{split}_features.parquet.

    python -m src.matching.features train test
"""
import sys

import pandas as pd
from rapidfuzz import fuzz

from ..common.io_utils import DATA, load_ground_truth, load_sources
from ..common.split import side_of


def load_pairs(split):
    """Candidate pairs; in train, pairs that cross the fit/val boundary are dropped."""
    pairs = pd.read_parquet(DATA / "candidates" / f"{split}_candidates.parquet")[["s1_id", "cand_id"]]
    if split == "train":
        side = side_of(set(pairs.s1_id) | set(pairs.cand_id))
        pairs = pairs[pairs.s1_id.map(side) == pairs.cand_id.map(side)].copy()
        pairs["side"] = pairs.s1_id.map(side)
    return pairs.reset_index(drop=True)


def build(split):
    src = load_sources(split)
    s1 = src["source1"].set_index("entity_id")
    other = pd.concat([src["source2"], src["source3"]]).set_index("entity_id")
    df = load_pairs(split)
    n1, n2 = s1.loc[df.s1_id, "business_name"].str.lower().values, other.loc[df.cand_id, "business_name"].str.lower().values
    a1, a2 = s1.loc[df.s1_id, "business_address"].str.lower().values, other.loc[df.cand_id, "business_address"].str.lower().values
    df["name_ratio"] = [fuzz.ratio(a, b) for a, b in zip(n1, n2)]
    df["name_token_set"] = [fuzz.token_set_ratio(a, b) for a, b in zip(n1, n2)]
    df["addr_ratio"] = [fuzz.ratio(a, b) for a, b in zip(a1, a2)]
    df["addr_token_set"] = [fuzz.token_set_ratio(a, b) for a, b in zip(a1, a2)]
    if split == "train":
        gt = load_ground_truth()
        truth = {(s, m) for s, ms in gt.items() for m in ms}
        df["label"] = [int((s, c) in truth) for s, c in zip(df.s1_id, df.cand_id)]
    out = DATA / "features"
    out.mkdir(parents=True, exist_ok=True)
    df.to_parquet(out / f"{split}_features.parquet", index=False)
    return df


if __name__ == "__main__":
    for split in sys.argv[1:] or ("train", "test"):
        print(split, build(split).shape)
