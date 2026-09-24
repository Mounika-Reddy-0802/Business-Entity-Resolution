"""One-change-at-a-time LightGBM search, judged on validation F0.5 after the decision sweep.

Each variant changes a single parameter from train_lgbm.PARAMS, retrains both stages on the fit
side, re-runs the decision sweep and logs a row in benchmarks/experiments.md. The winning value is
copied into PARAMS by hand, so the committed code always holds the chosen configuration.

    python -m src.matching.tune [param=value ...]    # default grid when no arguments
"""
import sys

from ..common.evaluate import log_run
from .decide import evaluate_val, sweep
from .train_lgbm import PARAMS, fit, predict

GRID = {"num_leaves": [15, 31, 127], "min_child_samples": [5, 50, 200],
        "scale_pos_weight": [0.5, 2.0], "learning_rate": [0.02, 0.1],
        "feature_fraction": [0.5, 1.0], "lambda_l2": [0.0, 10.0]}


def run(overrides):
    """Train, predict, sweep and log one variant; returns validation F0.5."""
    train = fit(**overrides)
    predict()
    cfg, _ = sweep()
    metrics = {**evaluate_val(cfg), "train": train, "decision": cfg}
    change = "tried: " + (", ".join(f"{k}={v}" for k, v in overrides.items()) or "committed params")
    log_run("tune", change, metrics)
    return metrics["f05"]


def parse(args):
    out = {}
    for a in args:
        k, v = a.split("=")
        out[k] = type(PARAMS[k])(float(v)) if isinstance(PARAMS[k], (int, float)) else v
    return out


def main(args):
    variants = [parse(args)] if args else [{k: v} for k, vals in GRID.items() for v in vals]
    results = [(run(v), v) for v in variants]
    fit()                                  # leave the committed configuration's model in place
    predict()
    sweep()
    for f, v in sorted(results, key=lambda r: -r[0]):
        print(f"{f:.4f} {v}")


if __name__ == "__main__":
    main(sys.argv[1:])
