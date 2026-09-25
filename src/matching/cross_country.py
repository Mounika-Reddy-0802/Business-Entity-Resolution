"""Cross-country transfer check, the proxy for the unseen test country (PLAN.md §1.5).

For every ordered pair of training countries (A, B): train on A's fit-side pairs, score B's
validation side, apply the current decision config and report macro F0.5 on B. Then drop one
feature group at a time and report the same number, to find features that do not transfer.

    python -m src.matching.cross_country [--log]
"""
import sys

import lightgbm as lgb
import pandas as pd

from ..blocking.block import KEYS
from ..blocking.normalise import load_normalised
from ..common.evaluate import log_run
from .decide import apply, fast_f05, load_config, sample_truth, truth_frame
from .features import load_features
from .train_lgbm import DROPPED, feature_cols, params

ROUNDS = 200
GROUPS = {
    "key_flags": lambda c: c in KEYS,
    "name_strings": lambda c: c.startswith("name_") and "_rank_" not in c and "_gap_" not in c
    and "_margin_" not in c and "_other_" not in c,
    "suffix": lambda c: c == "suffix_state",
    "addr_strings": lambda c: c.startswith("addr_") and "_rank_" not in c and "_gap_" not in c
    and "_margin_" not in c and "_other_" not in c,
    "numbers_postal": lambda c: c in {"house_state", "postal_state", "postal_prefix3_eq", "addr_num_jaccard"},
    "city": lambda c: c == "city_jaccard",
    "cross_field": lambda c: c in {"name1_in_addr2", "name2_in_addr1", "both_char_cos", "landmark_any"},
    "competition": lambda c: any(t in c for t in ("_rank_", "_gap_", "_margin_", "_other_s1_", "n_cands_s1", "n_s1_for_cand")),
    "source": lambda c: c == "is_s3",
}


def score_transfer(df, country, cols, train_c, test_c, cfg):
    """Macro F0.5 on test_c's validation entities for a model trained on train_c's fit side."""
    tr = df[(df.side == "fit") & (df.country == train_c)]
    te = df[(df.side == "val") & (df.country == test_c)].copy()
    model = lgb.train(params(), lgb.Dataset(tr[cols], tr.label), ROUNDS)
    te["p"] = model.predict(te[cols])
    truth = {k: v for k, v in sample_truth("val").items() if country.get(k) == test_c}
    return fast_f05(apply(te, cfg), truth_frame(truth), list(truth))


def main(log=False):
    df = load_features("train")
    s1 = load_normalised("train", ["entity_id", "country"])["source1"]
    country = dict(zip(s1.entity_id, s1.country))
    df["country"] = df.s1_id.map(country)
    cols = feature_cols(df.drop(columns="country"), DROPPED)
    cfg = load_config()
    countries = sorted(df.country.unique())
    res = {}
    for a in countries:
        for b in countries:
            res[f"{a}->{b}"] = score_transfer(df, country, cols, a, b, cfg)
            print(f"{a}->{b}: {res[f'{a}->{b}']:.4f}")
    cross = [k for k in res if k.split("->")[0] != k.split("->")[1]]
    ablation = {}
    for g, pred in GROUPS.items():
        keep = [c for c in cols if not pred(c)]
        if len(keep) == len(cols):
            continue
        ablation[g] = {k: score_transfer(df, country, keep, *k.split("->"), cfg) for k in cross}
        print(f"without {g}: " + ", ".join(f"{k} {v:.4f} ({v - res[k]:+.4f})" for k, v in ablation[g].items()))
    metrics = {**res, "cross_country": min(res[k] for k in cross), "ablation": ablation}
    if log:
        print(log_run("cross_country", "train one country, score the other; feature-group ablation",
                      metrics))
    return metrics


if __name__ == "__main__":
    main(log="--log" in sys.argv)
