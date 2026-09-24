"""LightGBM pair classifier.

    python -m src.matching.train_lgbm            # grouped CV OOF on the fit side, model on fit
    python -m src.matching.train_lgbm --predict  # scores for val side and test
"""
import sys

import lightgbm as lgb
import pandas as pd

from ..common.io_utils import DATA, ROOT

META = ["s1_id", "cand_id", "side", "label"]
PARAMS = {"objective": "binary", "learning_rate": 0.05, "num_leaves": 31, "min_child_samples": 20,
          "seed": 42, "verbose": -1, "num_threads": 0}
ROUNDS = 300


def feature_cols(df):
    return [c for c in df.columns if c not in META]


def fit():
    df = pd.read_parquet(DATA / "features" / "train_features.parquet")
    tr = df[df.side == "fit"]
    model = lgb.train(PARAMS, lgb.Dataset(tr[feature_cols(tr)], tr.label), ROUNDS)
    (ROOT / "models").mkdir(exist_ok=True)
    model.save_model(str(ROOT / "models" / "lgbm.txt"))
    return model


def predict():
    model = lgb.Booster(model_file=str(ROOT / "models" / "lgbm.txt"))
    out = DATA / "scores"
    out.mkdir(parents=True, exist_ok=True)
    tr = pd.read_parquet(DATA / "features" / "train_features.parquet")
    val = tr[tr.side == "val"].copy()
    val["p"] = model.predict(val[model.feature_name()])
    val[["s1_id", "cand_id", "p"]].to_parquet(out / "val_scores.parquet", index=False)
    te = pd.read_parquet(DATA / "features" / "test_features.parquet")
    te["p"] = model.predict(te[model.feature_name()])
    te[["s1_id", "cand_id", "p"]].to_parquet(out / "test_scores.parquet", index=False)


if __name__ == "__main__":
    predict() if "--predict" in sys.argv else fit()
