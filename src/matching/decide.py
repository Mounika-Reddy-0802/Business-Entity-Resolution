"""Decision layer (PLAN.md §2.5): pair scores -> match lists -> the two submission files.

Rules, applied in this order when enabled in the config:
  1-2  threshold per source (t_s2, t_s3; equal values = one global threshold)
  3    relative rule: keep p >= alpha * best p of the S1 entity
  4    one-to-one: an S2/S3 record goes only to the S1 entity where it scores highest
  5    cardinality caps: at most cap_s2 / cap_s3 matches per S1 entity
  6    singleton guard: empty list when the best p of the S1 entity is below t_single
  7    sanity: candidate ids only, no S1 ids, no duplicates (write_pairs_tsv + check_submission)

    python -m src.matching.decide sweep [--log]   # tune rules on OOF, keep those that lift val F0.5
    python -m src.matching.decide test [--log <tag> "<change>"]   # apply models/decision.json
"""
import json
import sys

import numpy as np
import pandas as pd

from ..blocking.normalise import load_normalised
from ..common.evaluate import blocking_recall, log_run, macro_f05, per_group_f05
from ..common.io_utils import DATA, OUTPUT, ROOT, load_ground_truth, pairs_to_map, write_pairs_tsv
from ..common.split import load_split

CONFIG = ROOT / "models" / "decision.json"
# one-to-one is on from the start: in the training truth no S2/S3 id belongs to two S1 entities,
# and the validation sample holds too few competing S1 entities to show its full effect
BASE = {"t_s2": 0.5, "t_s3": 0.5, "alpha": 0.0, "one_to_one": True, "cap_s2": 0, "cap_s3": 0,
        "t_single": 0.0}


def load_config():
    return {**BASE, **json.loads(CONFIG.read_text())} if CONFIG.exists() else dict(BASE)


def apply(scores, cfg):
    """Selected (s1_id, cand_id, p) rows after the enabled rules."""
    df = scores[["s1_id", "cand_id", "p"]]
    is_s3 = df.cand_id.str.startswith("S3-")
    best = df.groupby("s1_id").p.transform("max")
    keep = df.p >= np.where(is_s3, cfg["t_s3"], cfg["t_s2"])
    if cfg["alpha"] > 0:
        keep &= df.p >= cfg["alpha"] * best
    if cfg["one_to_one"]:
        keep &= df.p >= df.groupby("cand_id").p.transform("max")
    if cfg["t_single"] > 0:
        keep &= best >= cfg["t_single"]
    sel = df[keep]
    for src, cap in (("S2-", cfg["cap_s2"]), ("S3-", cfg["cap_s3"])):
        if cap > 0:
            part = sel.cand_id.str.startswith(src)
            rank = sel[part].groupby("s1_id").p.rank(ascending=False, method="first")
            sel = sel.drop(rank[rank > cap].index)
    return sel


def fast_f05(sel, truth_pairs, s1_ids):
    """Macro F0.5 over s1_ids from selected pairs and a truth pair frame (vectorised)."""
    tp = sel.merge(truth_pairs, on=["s1_id", "cand_id"]).groupby("s1_id").size()
    n_pred = sel.groupby("s1_id").size()
    n_true = truth_pairs.groupby("s1_id").size()
    idx = pd.Index(s1_ids)
    tp, n_pred, n_true = (x.reindex(idx, fill_value=0).to_numpy(float) for x in (tp, n_pred, n_true))
    p = np.divide(tp, n_pred, out=np.zeros_like(tp), where=n_pred > 0)
    r = np.divide(tp, n_true, out=np.zeros_like(tp), where=n_true > 0)
    f = np.divide(1.25 * p * r, 0.25 * p + r, out=np.zeros_like(tp), where=(0.25 * p + r) > 0)
    f[(n_true == 0) & (n_pred == 0)] = 1.0
    return float(f.mean())


def truth_frame(truth):
    return pd.DataFrame([(s, m) for s, ms in truth.items() for m in ms], columns=["s1_id", "cand_id"])


def learned_caps(truth):
    """99.5th percentile of S2 and S3 matches per S1 entity in the training truth."""
    per = pd.DataFrame([(sum(m.startswith("S2-") for m in v), sum(m.startswith("S3-") for m in v))
                        for v in truth.values()], columns=["s2", "s3"])
    return int(np.ceil(per.s2.quantile(0.995))), int(np.ceil(per.s3.quantile(0.995)))


def sample_truth(side):
    """Ground truth for the S1 entities of the fixed training sample on one side ('fit'/'val')."""
    from .features import train_sample
    gt = load_ground_truth()
    return {s: gt.get(s, []) for s, sd in train_sample().items() if sd == side}


def sweep(log=False):
    """Tune each rule on fit-side OOF scores in PLAN order; keep a rule only if val F0.5 rises."""
    oof = pd.read_parquet(DATA / "scores" / "train_oof.parquet")
    val = pd.read_parquet(DATA / "scores" / "val_scores.parquet")
    fit_truth, val_truth = sample_truth("fit"), sample_truth("val")
    fit_tp, val_tp = truth_frame(fit_truth), truth_frame(val_truth)
    fit_ids, val_ids = list(fit_truth), list(val_truth)
    f_oof = lambda c: fast_f05(apply(oof, c), fit_tp, fit_ids)
    f_val = lambda c: fast_f05(apply(val, c), val_tp, val_ids)

    cfg = dict(BASE)
    history = [{"rule": "t=0.5", "cfg": dict(cfg), "oof": f_oof(cfg), "val": f_val(cfg)}]
    grid_t = np.round(np.arange(0.30, 0.96, 0.025), 3)

    def try_rule(name, candidates):
        nonlocal cfg
        best = max(candidates, key=lambda c: f_oof({**cfg, **c}))
        new = {**cfg, **best}
        before, after = f_val(cfg), f_val(new)
        kept = after > before
        history.append({"rule": name, "tuned": best, "oof": f_oof(new), "val_before": before,
                        "val_after": after, "delta": after - before, "kept": kept})
        print(f"{name}: {best} val {before:.4f} -> {after:.4f} ({after - before:+.4f}) {'kept' if kept else 'dropped'}")
        if kept:
            cfg = new

    try_rule("1 global threshold", [{"t_s2": t, "t_s3": t} for t in grid_t])
    try_rule("2 per-source thresholds", [{"t_s2": a, "t_s3": b} for a in grid_t for b in grid_t])
    try_rule("3 relative alpha", [{"alpha": a} for a in np.round(np.arange(0.5, 0.96, 0.05), 2)])
    c2, c3 = learned_caps(fit_truth)
    try_rule("5 cardinality caps", [{"cap_s2": c2, "cap_s3": c3}])
    try_rule("6 singleton guard", [{"t_single": t} for t in np.round(np.arange(0.3, 0.99, 0.02), 2)])
    CONFIG.parent.mkdir(exist_ok=True)
    CONFIG.write_text(json.dumps(cfg, indent=2, default=float))
    print("final", cfg, f"val {f_val(cfg):.4f}")
    if log:
        metrics = {**evaluate_val(cfg), "decision": cfg, "sweep": history}
        print(log_run("decision_sweep", "decision rules 1-7 tuned on oof, kept if val f0.5 rises", metrics))
    return cfg, history


def evaluate_val(cfg=None):
    """Val F0.5 overall, per country and singleton/matched, plus val-side blocking recall."""
    cfg = cfg or load_config()
    scores = pd.read_parquet(DATA / "scores" / "val_scores.parquet")
    truth = sample_truth("val")
    s1 = load_normalised("train", ["entity_id", "country"])["source1"].set_index("entity_id")
    pred = pairs_to_map(apply(scores, cfg))
    country = {k: s1.loc[k, "country"] for k in truth}
    per = per_group_f05(pred, truth, country)
    single = {k: "singleton" if not v else "matched" for k, v in truth.items()}
    cands = pairs_to_map(scores)
    return {"f05": macro_f05(pred, truth), **{f"f05_{c}": v for c, v in per.items()},
            **{f"f05_{g}": v for g, v in per_group_f05(pred, truth, single).items()},
            "block_recall": blocking_recall(cands, truth),
            "cands_per_s1": sum(len(v) for v in cands.values()) / len(truth)}


def write_test(cfg=None):
    """Apply the decision config to test scores and write both submission files."""
    cfg = cfg or load_config()
    scores = pd.read_parquet(DATA / "scores" / "test_scores.parquet")
    s1_ids = list(load_normalised("test", ["entity_id"])["source1"].entity_id)
    write_pairs_tsv(apply(scores, cfg), s1_ids, OUTPUT / "matching_results.tsv", "matched_entity_ids")
    write_pairs_tsv(scores, s1_ids, OUTPUT / "candidate_pairs.tsv", "candidate_entity_ids")


if __name__ == "__main__":
    load_split()
    if len(sys.argv) > 1 and sys.argv[1] == "sweep":
        sweep(log="--log" in sys.argv)
        sys.exit(0)
    write_test()
    if not (DATA / "scores" / "val_scores.parquet").exists():
        print("model trained on the full training split; no validation score")
        sys.exit(0)
    metrics = evaluate_val()
    print(metrics)
    if "--log" in sys.argv:
        i = sys.argv.index("--log")
        train = json.loads((ROOT / "models" / "train_metrics.json").read_text())
        print(log_run(sys.argv[i + 1], sys.argv[i + 2], {**metrics, "train": train,
                                                          "decision": load_config()}))
