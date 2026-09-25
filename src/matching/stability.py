"""Seed stability of validation F0.5: retrain both stages with several LightGBM seeds on the fixed
split, re-run the decision sweep each time, and report mean and spread. A change is only worth
keeping when its gain is clearly larger than this spread.

    python -m src.matching.stability [n_seeds]
"""
import sys

import numpy as np

from ..common.evaluate import log_run
from .decide import evaluate_val, sweep
from .train_lgbm import fit, predict

SEEDS = [42, 7, 123, 2024, 99, 314, 777, 1001]


def main(n=5):
    scores = []
    for seed in SEEDS[:n]:
        fit(seed=seed)
        predict()
        cfg, _ = sweep()
        scores.append(evaluate_val(cfg)["f05"])
        print(f"seed {seed}: {scores[-1]:.4f}")
    fit()                                     # restore the committed seed's model and config
    predict()
    sweep()
    s = np.array(scores)
    metrics = {"f05": float(s.mean()), "f05_std": float(s.std(ddof=1)), "f05_min": float(s.min()),
               "f05_max": float(s.max()), "seeds": SEEDS[:n], "per_seed": scores}
    print(f"mean {s.mean():.4f} std {s.std(ddof=1):.4f} range {s.min():.4f}-{s.max():.4f}")
    log_run("stability", f"{n} lightgbm seeds, fixed split: mean val f0.5, std {s.std(ddof=1):.4f}",
            metrics)


if __name__ == "__main__":
    main(int(sys.argv[1]) if len(sys.argv) > 1 else 5)
