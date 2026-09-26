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
from rapidfuzz import fuzz, process
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GroupKFold

from ..common.io_utils import DATA, ROOT
from .features import SIBLING_TEXT, load_features, parts
from .ranking import second_largest

META = ["s1_id", "cand_id", "side", "label"]
PARAMS = {"objective": "binary", "learning_rate": 0.1, "num_leaves": 63, "min_child_samples": 20,
          "feature_fraction": 0.8, "bagging_fraction": 0.8, "bagging_freq": 1, "lambda_l2": 1.0,
          "scale_pos_weight": 1.0, "seed": 42, "deterministic": True, "verbose": -1,
          "num_threads": 0}
# removed after the cross-country check (benchmarks/experiments.md): dropping lifted both directions
DROPPED = ["is_s3"]
# second model on stage-1 score context within each S1 entity and sibling similarity (score_context)
STAGE2 = True
ANCHORS, ANCHOR_MIN = 3, 0.3
MAX_ROUNDS = 4000
FOLDS = 5
MODELS = ROOT / "models"
SCORES = DATA / "scores"


def params(**overrides):
    return {**PARAMS, **overrides}


def feature_cols(df, drop=()):
    """Model inputs: every column except ids, labels, dropped features and candidate texts."""
    return [c for c in df.columns if c not in META and c not in drop and c not in SIBLING_TEXT]


def train_rows(full):
    df = load_features("train")
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
    """Stage-2 features from stage-1 scores within each S1 entity (valid on a sample of entities,
    since an entity's whole candidate list is always featurised together):
    - rank, gap and margin of the pair's p among the entity's candidates;
    - sibling similarity: how much the candidate resembles the entity's confident matches
      (top ANCHORS candidates by p1 with p1 >= ANCHOR_MIN), on name skeleton, address skeleton,
      address numbers and the raw name. Records of one business resemble each other even when
      they differ from the S1 record (native script, renamed, moved)."""
    k = pd.factorize(df.s1_id)[0]
    p = np.asarray(p, dtype=np.float64)
    g = pd.Series(p).groupby(k)
    best, sec = g.transform("max").to_numpy(), second_largest(p, k)
    out = pd.DataFrame({
        "p1": p, "p1_rank_s1": g.rank(ascending=False, method="min").to_numpy(),
        "p1_gap_s1": best - p, "p1_margin_s1": np.where(p >= best, p - sec, p - best),
        "p1_sum_s1": g.transform("sum").to_numpy(),
        "p1_n_above_half_s1": pd.Series(p >= 0.5).groupby(k).transform("sum").to_numpy()}, index=df.index)
    rank = g.rank(ascending=False, method="first").to_numpy()
    rows = np.arange(len(df))
    anc = np.flatnonzero((rank <= ANCHORS) & (p >= ANCHOR_MIN))
    left = pd.DataFrame({"k": k, "row": rows})
    right = pd.DataFrame({"k": k[anc], "arow": anc, "ap": p[anc]})
    m = left.merge(right, on="k")
    m = m[m.row.to_numpy() != m.arow.to_numpy()]
    i, j = m.row.to_numpy(), m.arow.to_numpy()
    sim = {}
    for name, col, scorer in (("sib_name_sim", "sib_name", fuzz.token_set_ratio),
                              ("sib_addr_sim", "sib_addr", fuzz.token_set_ratio),
                              ("sib_num_sim", "sib_num", fuzz.token_set_ratio),
                              ("sib_raw_sim", "sib_raw", fuzz.ratio)):
        text = df[col].to_numpy(dtype=object)
        sim[name] = process.cpdist(text[i], text[j], scorer=scorer, workers=-1) / 100.0
    s_all = pd.DataFrame(sim)
    s_all["sib_pair_sim"] = (s_all.sib_name_sim + s_all.sib_addr_sim) / 2
    s_all["sib_weighted"] = s_all.sib_pair_sim * m.ap.to_numpy()
    s_all["row"] = i
    agg = s_all.groupby("row").max()
    agg["sib_n_anchors"] = s_all.groupby("row").size()
    out = out.join(agg.reindex(rows).set_axis(df.index))
    out["sib_n_anchors"] = out.sib_n_anchors.fillna(0)
    return out.astype(np.float32)


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
    """Score the test split, and the validation side unless the model was trained on it."""
    SCORES.mkdir(parents=True, exist_ok=True)
    full = json.loads((MODELS / "train_metrics.json").read_text()).get("full", False)
    if full:
        (SCORES / "val_scores.parquet").unlink(missing_ok=True)
    else:
        tr = load_features("train")
        val = tr[tr.side == "val"].reset_index(drop=True)
        val["p"] = score(val)
        val[["s1_id", "cand_id", "p"]].to_parquet(SCORES / "val_scores.parquet", index=False)
    scored = []
    for f in parts("test"):                     # tens of millions of pairs: one part at a time
        te = pd.read_parquet(f)
        scored.append(te[["s1_id", "cand_id"]].assign(p=score(te)))
    pd.concat(scored, ignore_index=True).to_parquet(SCORES / "test_scores.parquet", index=False)


if __name__ == "__main__":
    if "--predict" in sys.argv:
        predict()
    else:
        print(fit(full="--full" in sys.argv))
