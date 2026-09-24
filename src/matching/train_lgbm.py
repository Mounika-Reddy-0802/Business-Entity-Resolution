"""LightGBM pair classifier with grouped cross-validation.

    python -m src.matching.train_lgbm [--full]   # 5-fold GroupKFold OOF + final model, 2 stages
    python -m src.matching.train_lgbm --predict  # scores for the val side and for test

Default: trains on the fit side of the training split, so the validation side stays unseen.
--full: trains on the whole training split (final submission); OOF then covers every train pair.
"""
import json
import sys

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GroupKFold

from ..common.io_utils import DATA, ROOT
from .features import second_largest

META = ["s1_id", "cand_id", "side", "label"]
PARAMS = {"objective": "binary", "learning_rate": 0.05, "num_leaves": 63, "min_child_samples": 20,
          "feature_fraction": 0.8, "bagging_fraction": 0.8, "bagging_freq": 1, "lambda_l2": 1.0,
          "scale_pos_weight": 1.0, "seed": 42, "deterministic": True, "verbose": -1,
          "num_threads": 0}
# removed after the cross-country check (benchmarks/experiments.md): dropping lifted both directions
DROPPED = ["is_s3"]
STAGE2 = True             # second model on stage-1 score context (see score_context)
MAX_ROUNDS = 2000
FOLDS = 5
MODELS = ROOT / "models"
SCORES = DATA / "scores"


def params(**overrides):
    return {**PARAMS, **overrides}


def feature_cols(df, drop=()):
    return [c for c in df.columns if c not in META and c not in drop]


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


def score_context(df, p):
    """Stage-2 features from stage-1 scores: how a pair's p compares with the other candidates of
    its S1 entity and with the other S1 entities that list the same S2/S3 record."""
    ctx = pd.DataFrame({"s1_id": df.s1_id.values, "cand_id": df.cand_id.values, "p1": p})
    g1, g2 = ctx.groupby("s1_id").p1, ctx.groupby("cand_id").p1
    best1, sec1 = g1.transform("max"), second_largest(ctx.p1, ctx.s1_id)
    best2, sec2 = g2.transform("max"), second_largest(ctx.p1, ctx.cand_id)
    other = np.where(ctx.p1 >= best2, sec2, best2)
    return pd.DataFrame({
        "p1": ctx.p1, "p1_rank_s1": g1.rank(ascending=False, method="min"),
        "p1_gap_s1": best1 - ctx.p1, "p1_margin_s1": np.where(ctx.p1 >= best1, ctx.p1 - sec1, ctx.p1 - best1),
        "p1_sum_s1": g1.transform("sum"), "p1_n_above_half_s1": (ctx.p1 >= 0.5).groupby(ctx.s1_id).transform("sum"),
        "p1_other_s1_best": other, "p1_margin_cand": ctx.p1 - other,
        "p1_rank_cand": g2.rank(ascending=False, method="min"),
    }, index=df.index).astype(np.float32)


def fit_stage(df, cols, prm):
    """Grouped OOF scores and a final model for one stage."""
    oof, rounds = cross_validate(df, cols, prm)
    n_rounds = int(np.mean(rounds) * 1.1)
    return oof, lgb.train(prm, lgb.Dataset(df[cols], df.label), n_rounds), n_rounds


def fit(full=False, **overrides):
    """Two stages of grouped CV + final model; OOF of the last stage feeds the decision sweep.
    overrides: LightGBM params."""
    df = train_rows(full)
    cols = feature_cols(df, DROPPED)
    prm = params(**overrides)
    oof1, model1, rounds1 = fit_stage(df, cols, prm)
    MODELS.mkdir(exist_ok=True)
    model1.save_model(str(MODELS / "lgbm.txt"))
    oof, rounds = oof1, rounds1
    if STAGE2:
        ctx = score_context(df, oof1)
        df2 = pd.concat([df, ctx], axis=1)
        oof, model2, rounds = fit_stage(df2, cols + list(ctx.columns), prm)
        model2.save_model(str(MODELS / "lgbm_stage2.txt"))
        model = model2
    else:
        (MODELS / "lgbm_stage2.txt").unlink(missing_ok=True)
        model = model1
    SCORES.mkdir(parents=True, exist_ok=True)
    df[["s1_id", "cand_id"]].assign(p=oof, label=df.label).to_parquet(SCORES / "train_oof.parquet", index=False)
    imp = pd.DataFrame({"feature": model.feature_name(), "gain": model.feature_importance("gain")}
                       ).sort_values("gain", ascending=False)
    imp.to_csv(ROOT / "benchmarks" / "raw" / "feature_importance.tsv", sep="	", index=False)
    metrics = {"oof_auc": float(roc_auc_score(df.label, oof)),
               "oof_auc_stage1": float(roc_auc_score(df.label, oof1)), "rounds": rounds,
               "n_features": len(model.feature_name()), "train_pairs": len(df),
               "train_pos": int(df.label.sum()), "full": full, "stage2": STAGE2, "params": prm}
    (MODELS / "train_metrics.json").write_text(json.dumps(metrics, indent=2))
    return metrics


def score(df):
    """Final probabilities for a feature frame with the saved model(s)."""
    model1 = lgb.Booster(model_file=str(MODELS / "lgbm.txt"))
    p = model1.predict(df[model1.feature_name()])
    stage2 = MODELS / "lgbm_stage2.txt"
    if stage2.exists():
        model2 = lgb.Booster(model_file=str(stage2))
        df2 = pd.concat([df, score_context(df, p)], axis=1)
        p = model2.predict(df2[model2.feature_name()])
    return p


def predict():
    """Score the validation side and the test split."""
    SCORES.mkdir(parents=True, exist_ok=True)
    tr = pd.read_parquet(DATA / "features" / "train_features.parquet")
    val = tr[tr.side == "val"].reset_index(drop=True)
    val["p"] = score(val)
    val[["s1_id", "cand_id", "p"]].to_parquet(SCORES / "val_scores.parquet", index=False)
    te = pd.read_parquet(DATA / "features" / "test_features.parquet")
    te["p"] = score(te)
    te[["s1_id", "cand_id", "p"]].to_parquet(SCORES / "test_scores.parquet", index=False)


if __name__ == "__main__":
    if "--predict" in sys.argv:
        predict()
    else:
        print(fit(full="--full" in sys.argv))
