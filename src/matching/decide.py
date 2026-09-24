"""Turn pair scores into match lists and write the two submission files.

    python -m src.matching.decide test [--log <tag> "<change>"]
"""
import json
import sys

import pandas as pd

from ..common.evaluate import blocking_recall, log_run, macro_f05, per_group_f05
from ..common.io_utils import DATA, OUTPUT, ROOT, load_sources, pairs_to_map, write_id_list_tsv
from ..common.split import ground_truth_for, load_split

THRESHOLD = 0.5


def decide(scores, t=THRESHOLD):
    """{s1_id: [cand ids with p >= t]}."""
    keep = scores[scores.p >= t]
    return pairs_to_map(keep)


def evaluate_val():
    """Val F0.5 overall and per country, plus blocking recall on the val side."""
    scores = pd.read_parquet(DATA / "scores" / "val_scores.parquet")
    truth = ground_truth_for("val")
    s1 = load_sources("train")["source1"].set_index("entity_id")
    pred = decide(scores)
    country = {k: s1.loc[k, "country"] for k in truth}
    per = per_group_f05(pred, truth, country)
    cands = pairs_to_map(scores)
    return {"f05": macro_f05(pred, truth), **{f"f05_{c}": v for c, v in per.items()},
            "block_recall": blocking_recall(cands, truth),
            "cands_per_s1": sum(len(v) for v in cands.values()) / len(truth)}


def write_test():
    scores = pd.read_parquet(DATA / "scores" / "test_scores.parquet")
    s1_ids = list(load_sources("test")["source1"]["entity_id"])
    write_id_list_tsv(decide(scores), s1_ids, OUTPUT / "matching_results.tsv", "matched_entity_ids")
    write_id_list_tsv(pairs_to_map(scores), s1_ids, OUTPUT / "candidate_pairs.tsv", "candidate_entity_ids")


if __name__ == "__main__":
    load_split()
    metrics = evaluate_val()
    print(metrics)
    write_test()
    if "--log" in sys.argv:
        i = sys.argv.index("--log")
        train = json.loads((ROOT / "models" / "train_metrics.json").read_text())
        print(log_run(sys.argv[i + 1], sys.argv[i + 2], {**metrics, "train": train}))
