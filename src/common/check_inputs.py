"""Checks the organiser files before anything else runs, so a surprise in the real data fails
loudly on day one instead of silently degrading a later stage.

Checks: expected columns, id prefixes match the file, no duplicate ids, no empty ids, country
labels per split (a label only in test is reported, not rejected), ground-truth ids exist and
belong to S2/S3, no S2/S3 record matched to two S1 entities, share of empty names/addresses.

    python -m src.common.check_inputs
"""
import csv
import sys
from collections import Counter

import pandas as pd

from .io_utils import RAW, SOURCES, data_is_synthetic, load_tsv, parse_id_list

COLUMNS = ["entity_id", "business_name", "business_address", "country"]


def read_source(split, source):
    """One source file with Arrow strings (millions of rows stay affordable)."""
    return pd.read_csv(RAW / split / f"{split}_{source}.tsv", sep="\t", dtype="string[pyarrow]",
                       keep_default_na=False, quoting=csv.QUOTE_NONE, encoding="utf-8-sig")


def check_split(split):
    """(problems, facts, countries, ids) for one split, one source file at a time."""
    problems, facts, countries, ids = [], {}, set(), {}
    for s in SOURCES:
        df = read_source(split, s)
        prefix = "S" + s[-1] + "-"
        if list(df.columns) != COLUMNS:
            problems.append(f"{split}_{s}: columns {list(df.columns)} != {COLUMNS}")
            continue
        bad = (~df.entity_id.str.startswith(prefix)).sum()
        if bad:
            problems.append(f"{split}_{s}: {bad} ids without prefix {prefix}")
        dup = df.entity_id.duplicated().sum()
        if dup:
            problems.append(f"{split}_{s}: {dup} duplicate ids")
        facts[f"{split}_{s}"] = {
            "rows": len(df), "countries": dict(Counter(df.country)),
            "empty_name": round(float((df.business_name.str.strip() == "").mean()), 4),
            "empty_address": round(float((df.business_address.str.strip() == "").mean()), 4)}
        countries |= set(df.country.unique())
        ids[s] = set(df.entity_id) if split == "train" else None
    return problems, facts, countries, ids


def check_truth(ids):
    """Problems in train_ground_truth.tsv against the training source ids."""
    problems = []
    gt = load_tsv(RAW / "train" / "train_ground_truth.tsv")
    s1 = ids["source1"]
    other = ids["source2"] | ids["source3"]
    missing_rows = s1 - set(gt.source1_entity_id)
    if missing_rows:
        problems.append(f"ground truth lacks {len(missing_rows)} S1 entities (treated as singletons)")
    matched = [m for cell in gt.matched_entity_ids for m in parse_id_list(cell)]
    unknown = [m for m in matched if m not in other]
    if unknown:
        problems.append(f"ground truth lists {len(unknown)} ids not in train S2/S3, e.g. {unknown[:3]}")
    twice = [m for m, k in Counter(matched).items() if k > 1]
    if twice:
        problems.append(f"{len(twice)} S2/S3 ids matched to more than one S1 entity "
                        "(one-to-one assumption in docs/decisions.md needs review)")
    return problems


def main():
    problems, facts, found = [], {}, {}
    for split in ("train", "test"):
        p, f, found[split], ids = check_split(split)
        problems += p
        facts.update(f)
        if split == "train":
            problems += check_truth(ids)
            del ids
    train_c, test_c = found["train"], found["test"]
    print("synthetic stand-in data" if data_is_synthetic() else "organiser data")
    for k, v in facts.items():
        print(f"{k}: {v}")
    print(f"countries only in test: {sorted(test_c - train_c)}; only in train: {sorted(train_c - test_c)}")
    for p in problems:
        print("PROBLEM:", p)
    sys.exit(1 if any("columns" in p or "prefix" in p or "duplicate" in p for p in problems) else 0)


if __name__ == "__main__":
    main()
