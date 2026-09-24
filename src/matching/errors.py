"""Dump the worst validation entities for error analysis -> benchmarks/raw/errors_<timestamp>.tsv.

One row per wrong pair: a miss (true match not predicted; `stage` says whether blocking or the
model lost it) or a false merge (predicted, not true). Entities are ordered by their F0.5.

    python -m src.matching.errors [n_entities]
"""
import sys
import time

import pandas as pd

from ..blocking.normalise import load_normalised
from ..common.evaluate import f05_entity
from ..common.io_utils import DATA, ROOT, pairs_to_map
from ..common.split import ground_truth_for
from .decide import apply, load_config


def main(n=200):
    val = pd.read_parquet(DATA / "scores" / "val_scores.parquet")
    truth = ground_truth_for("val")
    pred = pairs_to_map(apply(val, load_config()))
    cands = pairs_to_map(val)
    p = {(s, c): v for s, c, v in zip(val.s1_id, val.cand_id, val.p)}
    src = load_normalised("train")
    recs = pd.concat(src.values()).set_index("entity_id")
    worst = sorted(((f05_entity(pred.get(k, []), v), k) for k, v in truth.items()))[:n]
    rows = []
    for f, s in worst:
        if f == 1.0:
            break
        t, pr, cs = set(truth[s]), set(pred.get(s, [])), set(cands.get(s, []))
        for c in sorted((t - pr) | (pr - t)):
            kind = "miss" if c in t else "false_merge"
            stage = "" if kind == "false_merge" else ("blocking" if c not in cs else "model")
            rows.append({"s1_id": s, "entity_f05": round(f, 3), "kind": kind, "stage": stage,
                         "cand_id": c, "p": round(p.get((s, c), float("nan")), 4),
                         "n_true": len(t), "country": recs.loc[s, "country"],
                         "s1_name": recs.loc[s, "business_name"], "cand_name": recs.loc[c, "business_name"],
                         "s1_addr": recs.loc[s, "business_address"], "cand_addr": recs.loc[c, "business_address"]})
    df = pd.DataFrame(rows)
    path = ROOT / "benchmarks" / "raw" / f"errors_{time.strftime('%Y%m%d_%H%M%S')}.tsv"
    df.to_csv(path, sep="\t", index=False)
    summary = df.groupby(["kind", "stage"]).size()
    print(summary.to_string())
    print(path)
    return df


if __name__ == "__main__":
    main(int(sys.argv[1]) if len(sys.argv) > 1 else 200)
