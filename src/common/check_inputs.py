"""Checks the organiser files before anything else runs, so a surprise in the real data fails
loudly on day one instead of silently degrading a later stage.

Checks: expected columns, id prefixes match the file, no duplicate ids, no empty ids, country
labels per split (a label only in test is reported, not rejected), ground-truth ids exist and
belong to S2/S3, no S2/S3 record matched to two S1 entities, share of empty names/addresses.

    python -m src.common.check_inputs
"""
import sys
from collections import Counter

from .io_utils import RAW, data_is_synthetic, load_sources, load_tsv, parse_id_list

COLUMNS = ["entity_id", "business_name", "business_address", "country"]


def check_split(split, src):
    """(problems, facts) for one split's three source frames."""
    problems, facts = [], {}
    for s, df in src.items():
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
    return problems, facts


def check_truth(train):
    """Problems in train_ground_truth.tsv against the training sources."""
    problems = []
    gt = load_tsv(RAW / "train" / "train_ground_truth.tsv")
    s1 = set(train["source1"].entity_id)
    other = set(train["source2"].entity_id) | set(train["source3"].entity_id)
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
    train, test = load_sources("train"), load_sources("test")
    problems, facts = [], {}
    for split, src in (("train", train), ("test", test)):
        p, f = check_split(split, src)
        problems += p
        facts.update(f)
    problems += check_truth(train)
    train_c = {c for df in train.values() for c in df.country}
    test_c = {c for df in test.values() for c in df.country}
    print("synthetic stand-in data" if data_is_synthetic() else "organiser data")
    for k, v in facts.items():
        print(f"{k}: {v}")
    print(f"countries only in test: {sorted(test_c - train_c)}; only in train: {sorted(train_c - test_c)}")
    for p in problems:
        print("PROBLEM:", p)
    sys.exit(1 if any("columns" in p or "prefix" in p or "duplicate" in p for p in problems) else 0)


if __name__ == "__main__":
    main()
