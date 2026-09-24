"""Pre-upload checks on the test outputs (PLAN.md §5) -> benchmarks/raw/<timestamp>_sanity.json.

Row count equals the test S1 count, every matched id is a candidate of the same entity, and the
predicted matches-per-entity distribution per country is printed next to the training truth so an
unseen country (France) that behaves very differently stands out.

    python -m src.matching.sanity
"""
import json
import sys
import time

import pandas as pd

from ..common.io_utils import OUTPUT, ROOT, load_ground_truth, load_sources, read_id_list_tsv


def distribution(counts):
    """Share of entities with 0, 1, 2, 3+ matches and the mean."""
    s = pd.Series(counts)
    return {"n": int(len(s)), "mean": round(float(s.mean()), 3),
            **{f"share_{k}": round(float((s == k).mean()), 3) for k in (0, 1, 2)},
            "share_3plus": round(float((s >= 3).mean()), 3)}


def main():
    test = load_sources("test")["source1"]
    matches = read_id_list_tsv(OUTPUT / "matching_results.tsv", "matched_entity_ids")
    cands = read_id_list_tsv(OUTPUT / "candidate_pairs.tsv", "candidate_entity_ids")
    problems = []
    if len(matches) != len(test) or set(matches) != set(test.entity_id):
        problems.append(f"matching rows {len(matches)} vs test S1 {len(test)}")
    outside = sum(1 for s, ms in matches.items() for m in ms if m not in set(cands.get(s, [])))
    if outside:
        problems.append(f"{outside} matched ids are not candidates of their entity")
    train_s1 = load_sources("train")["source1"].set_index("entity_id")
    truth = load_ground_truth()
    report = {"problems": problems, "train_truth": {}, "test_predicted": {}}
    for country, ids in train_s1.groupby("country").groups.items():
        report["train_truth"][country] = distribution([len(truth.get(i, [])) for i in ids])
    for country, g in test.groupby("country"):
        report["test_predicted"][country] = distribution([len(matches.get(i, [])) for i in g.entity_id])
    report["train_truth"]["all"] = distribution([len(v) for v in truth.values()])
    report["test_predicted"]["all"] = distribution([len(v) for v in matches.values()])
    for side in ("train_truth", "test_predicted"):
        for country, d in report[side].items():
            print(f"{side:15s} {country:8s} {d}")
    path = ROOT / "benchmarks" / "raw" / f"{time.strftime('%Y%m%d_%H%M%S')}_sanity.json"
    path.write_text(json.dumps(report, indent=2) + "\n")
    print(path)
    if problems:
        print("\n".join(problems))
        sys.exit(1)


if __name__ == "__main__":
    main()
