"""Local submission check, applying every rule from the problem statement.

Stand-in while utils/validate_submission.py (organiser script) is not on this machine; once it is,
scripts/run_pipeline.sh runs the organiser script and this one is only a second opinion.

    python -m src.common.check_submission --matching output/matching_results.tsv \
        --candidate output/candidate_pairs.tsv --test-dir data/raw/dataset/test
"""
import argparse
import csv
import sys
from pathlib import Path


def read_rows(path, id_col):
    """Returns (header, rows) with rows as (s1_id, [ids]); raw strings, no pandas."""
    with open(path, newline="", encoding="utf-8") as f:
        reader = csv.reader(f, delimiter="\t")
        header = next(reader, [])
        rows = [(r[0] if r else "", [x for x in (r[1].split(",") if len(r) > 1 else []) if x != ""])
                for r in reader]
    return header, rows


def source_ids(test_dir):
    """{source name: set of entity ids} from the test source files."""
    out = {}
    for s in ("source1", "source2", "source3"):
        with open(Path(test_dir) / f"test_{s}.tsv", newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f, delimiter="\t")
            out[s] = {r["entity_id"] for r in reader}
    return out


def check_file(path, id_col, ids):
    """List of rule violations for one output file."""
    issues = []
    header, rows = read_rows(path, id_col)
    if header != ["source1_entity_id", id_col]:
        issues.append(f"{path}: header {header} != ['source1_entity_id', '{id_col}']")
    seen = [r[0] for r in rows]
    if len(seen) != len(set(seen)):
        issues.append(f"{path}: duplicate source1_entity_id rows")
    missing = ids["source1"] - set(seen)
    extra = set(seen) - ids["source1"]
    if missing:
        issues.append(f"{path}: {len(missing)} test S1 entities missing")
    if extra:
        issues.append(f"{path}: {len(extra)} rows are not test S1 entities")
    other = ids["source2"] | ids["source3"]
    for s1, lst in rows:
        if len(lst) != len(set(lst)):
            issues.append(f"{path}: duplicate ids in list for {s1}")
        bad = [x for x in lst if x not in other]
        if bad:
            issues.append(f"{path}: {s1} lists ids not in test S2/S3: {bad[:3]}")
    return issues, dict(rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--matching", required=True)
    ap.add_argument("--candidate", required=True)
    ap.add_argument("--test-dir", required=True)
    a = ap.parse_args()
    ids = source_ids(a.test_dir)
    issues_m, matches = check_file(a.matching, "matched_entity_ids", ids)
    issues_c, cands = check_file(a.candidate, "candidate_entity_ids", ids)
    issues = issues_m + issues_c
    not_cand = sum(1 for s, lst in matches.items() for x in lst if x not in set(cands.get(s, [])))
    if not_cand:
        issues.append(f"{not_cand} matched ids are not candidates of the same S1 entity")
    if issues:
        for i, msg in enumerate(issues, 1):
            print(f"{i}. {msg}")
        sys.exit(1)
    print("PASS")


if __name__ == "__main__":
    main()
