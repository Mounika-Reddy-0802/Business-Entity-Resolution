"""Official metric (macro F0.5 with the singleton rule) plus blocking diagnostics.
The worked example in the problem statement must give 0.714 (tests/test_evaluate.py)."""
from collections import Counter


def f05_entity(pred, true):
    pred, true = set(pred), set(true)
    if not true and not pred:
        return 1.0
    if not true or not pred:
        return 0.0
    tp = len(pred & true)
    if tp == 0:
        return 0.0
    p, r = tp / len(pred), tp / len(true)
    return 1.25 * p * r / (0.25 * p + r)


def macro_f05(pred_map, true_map):
    """true_map defines the evaluation set; a missing prediction counts as an empty list."""
    return sum(f05_entity(pred_map.get(k, []), v) for k, v in true_map.items()) / len(true_map)


def per_group_f05(pred_map, true_map, group_of):
    """Macro F0.5 per group (e.g. country, singleton flag). group_of: {s1_id: group}."""
    tot, n = Counter(), Counter()
    for k, v in true_map.items():
        g = group_of.get(k, "?")
        tot[g] += f05_entity(pred_map.get(k, []), v)
        n[g] += 1
    return {g: tot[g] / n[g] for g in tot}


def blocking_recall(cand_map, true_map):
    """Fraction of true (s1, match) pairs present in the candidate map."""
    hit = total = 0
    for k, v in true_map.items():
        c = set(cand_map.get(k, []))
        total += len(v)
        hit += sum(1 for m in v if m in c)
    return hit / total if total else 1.0


def candidate_stats(cand_map, n_s1, n_other):
    sizes = [len(v) for v in cand_map.values()]
    total = sum(sizes)
    return {
        "pairs": total,
        "mean_per_s1": total / n_s1 if n_s1 else 0.0,
        "max_per_s1": max(sizes) if sizes else 0,
        "reduction_ratio": 1 - total / (n_s1 * n_other) if n_s1 and n_other else 0.0,
    }


def git_commit():
    """Short HEAD hash; a trailing + marks uncommitted changes under src/ or scripts/."""
    import subprocess
    from .io_utils import ROOT
    run = lambda *a: subprocess.run(["git", *a], cwd=ROOT, capture_output=True, text=True).stdout.strip()
    head = run("rev-parse", "--short", "HEAD") or "none"
    return head + ("+" if run("status", "--porcelain", "--", "src", "scripts") else "")


def log_run(tag, change, metrics, member=None):
    """Write benchmarks/raw/<timestamp>_<tag>.json and append one row to experiments.md.
    metrics may hold block_recall, cands_per_s1, f05, f05_US, f05_India, cross_country, ..."""
    import json
    import subprocess
    import time
    from .io_utils import ROOT, data_is_synthetic
    stamp = time.strftime("%Y%m%d_%H%M%S")
    who = member or subprocess.run(["git", "config", "user.name"], cwd=ROOT, capture_output=True,
                                   text=True).stdout.strip()
    rec = {"time": time.strftime("%Y-%m-%d %H:%M"), "tag": tag, "change": change, "who": who,
           "commit": git_commit(), "synthetic_data": data_is_synthetic(), **metrics}
    raw = ROOT / "benchmarks" / "raw"
    raw.mkdir(parents=True, exist_ok=True)
    path = raw / f"{stamp}_{tag}.json"
    path.write_text(json.dumps(rec, indent=2, default=float) + "\n")
    fmt = lambda k: f"{metrics[k]:.4f}" if isinstance(metrics.get(k), float) else str(metrics.get(k, ""))
    change_txt = change + (" (synthetic)" if rec["synthetic_data"] else "")
    row = [time.strftime("%H:%M"), who, rec["commit"], change_txt, fmt("block_recall"),
           fmt("cands_per_s1"), fmt("f05"), fmt("f05_US"), fmt("f05_India"), fmt("cross_country"),
           "yes" if metrics.get("uploaded") else "no",
           "" if metrics.get("lb") is None else f"{metrics['lb']:.4f}"]
    with open(ROOT / "benchmarks" / "experiments.md", "a", encoding="utf-8") as f:
        f.write("| " + " | ".join(row) + " |\n")
    return path
