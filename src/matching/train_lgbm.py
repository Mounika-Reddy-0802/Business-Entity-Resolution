"""LightGBM pair classifier with grouped cross-validation.

    python -m src.matching.train_lgbm [--full]   # 5-fold GroupKFold OOF + final model
    python -m src.matching.train_lgbm --predict  # scores for the val side and for test

Default: trains on the fit side of the training split, so the validation side stays unseen.
--full: trains on the whole training split (final submission); OOF then covers every train pair.
Hyperparameters live in models/lgbm_params.json when present (written by tuning runs), otherwise
PARAMS below.
"""
import json
import sys

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GroupKFold

from ..common.io_utils import DATA, ROOT

META = ["s1_id", "cand_id", "side", "label"]
PARAMS = {"objective": "binary", "learning_rate": 0.05, "num_leaves": 63, "min_child_samples": 20,
          "feature_fraction": 0.8, "bagging_fraction": 0.8, "bagging_freq": 1, "lambda_l2": 1.0,
          "scale_pos_weight": 1.0, "seed": 42, "deterministic": True, "verbose": -1,
          "num_threads": 0}
MAX_ROUNDS = 2000
FOLDS = 5
MODELS = ROOT / "models"
SCORES = DATA / "scores"


def params():
    path = MODELS / "lgbm_params.json"
    return {**PARAMS, **json.loads(path.read_text())} if path.exists() else dict(PARAMS)


def feature_cols(df, drop=()):
    return [c for c in df.columns if c not in META and c not in drop]


def dropped_features():
    """Features removed by the cross-country check (models/dropped_features.json)."""
    path = MODELS / "dropped_features.json"
    return json.loads(path.read_text()) if path.exists() else []


def train_rows(full):
    df = pd.read_parquet(DATA / "features" / "train_features.parquet")
    return df if full else df[df.side == "fit"].reset_index(drop=True)


def cross_validate(df, cols, prm):
    """Grouped OOF probabilities and the best round of each fold."""
    oof = np.zeros(len(df))
    rounds = []
    for fold, (tr, va) in enumerate(GroupKFold(FOLDS).split(df, groups=df.s1_id)):
        dtr = lgb.Dataset(df.loc[tr, cols], df.label[tr])
        dva = lgb.Dataset(df.loc[va, cols], df.label[va])
        m = lgb.train(prm, dtr, MAX_ROUNDS, valid_sets=[dva],
                      callbacks=[lgb.early_stopping(100, verbose=False)])
        oof[va] = m.predict(df.loc[va, cols], num_iteration=m.best_iteration)
        rounds.append(m.best_iteration)
    return oof, rounds


def fit(full=False):
    """CV, OOF scores, final model, importance; returns a metrics dict."""
    df = train_rows(full)
    cols = feature_cols(df, dropped_features())
    prm = params()
    oof, rounds = cross_validate(df, cols, prm)
    SCORES.mkdir(parents=True, exist_ok=True)
    df[["s1_id", "cand_id"]].assign(p=oof, label=df.label).to_parquet(SCORES / "train_oof.parquet", index=False)
    n_rounds = int(np.mean(rounds) * 1.1)
    model = lgb.train(prm, lgb.Dataset(df[cols], df.label), n_rounds)
    MODELS.mkdir(exist_ok=True)
    model.save_model(str(MODELS / "lgbm.txt"))
    imp = pd.DataFrame({"feature": cols, "gain": model.feature_importance("gain")}).sort_values(
        "gain", ascending=False)
    imp.to_csv(ROOT / "benchmarks" / "raw" / "feature_importance.tsv", sep="\t", index=False)
    metrics = {"oof_auc": float(roc_auc_score(df.label, oof)), "rounds": n_rounds,
               "n_features": len(cols), "train_pairs": len(df), "train_pos": int(df.label.sum()),
               "full": full, "params": prm}
    (MODELS / "train_metrics.json").write_text(json.dumps(metrics, indent=2))
    return metrics


def predict():
    """Score the validation side and the test split with the saved model."""
    model = lgb.Booster(model_file=str(MODELS / "lgbm.txt"))
    cols = model.feature_name()
    SCORES.mkdir(parents=True, exist_ok=True)
    tr = pd.read_parquet(DATA / "features" / "train_features.parquet")
    val = tr[tr.side == "val"].copy()
    val["p"] = model.predict(val[cols])
    val[["s1_id", "cand_id", "p"]].to_parquet(SCORES / "val_scores.parquet", index=False)
    te = pd.read_parquet(DATA / "features" / "test_features.parquet")
    te["p"] = model.predict(te[cols])
    te[["s1_id", "cand_id", "p"]].to_parquet(SCORES / "test_scores.parquet", index=False)


if __name__ == "__main__":
    if "--predict" in sys.argv:
        predict()
    else:
        print(fit(full="--full" in sys.argv))
