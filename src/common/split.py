"""Fixed validation split. data/ is gitignored, so every laptop regenerates it with the same seed
and gets the identical split.

Validation entities are chosen from Source 1, stratified by country and by singleton/non-singleton.
Every S2/S3 record matched to a validation S1 entity belongs to the validation side; unmatched S2/S3
records are split at random with the same seed."""
import random
from collections import defaultdict

from .io_utils import DATA, load_ground_truth, load_sources

SEED = 42
VAL_FRACTION = 0.2
SPLIT_DIR = DATA / "splits"


def make_split(seed=SEED, val_fraction=VAL_FRACTION):
    src = load_sources("train")
    gt = load_ground_truth()
    strata = defaultdict(list)
    for r in src["source1"].itertuples():
        strata[(r.country, len(gt.get(r.entity_id, [])) == 0)].append(r.entity_id)
    rng = random.Random(seed)
    val_s1 = set()
    for key in sorted(strata):
        ids = sorted(strata[key])
        rng.shuffle(ids)
        val_s1.update(ids[: int(round(len(ids) * val_fraction))])
    matched_to_val = {m for s, ms in gt.items() if s in val_s1 for m in ms}
    matched_to_fit = {m for s, ms in gt.items() if s not in val_s1 for m in ms}
    val_other = set()
    for s in ("source2", "source3"):
        for eid in sorted(src[s]["entity_id"]):
            if eid in matched_to_val:
                val_other.add(eid)
            elif eid in matched_to_fit:
                continue
            elif rng.random() < val_fraction:
                val_other.add(eid)
    SPLIT_DIR.mkdir(parents=True, exist_ok=True)
    for name, ids in (("val_s1_ids.txt", val_s1), ("val_other_ids.txt", val_other)):
        path, text = SPLIT_DIR / name, "\n".join(sorted(ids)) + "\n"
        if not path.exists() or path.read_text() != text:    # unchanged files keep their mtime
            path.write_text(text)
    return val_s1, val_other


def load_split():
    if not (SPLIT_DIR / "val_s1_ids.txt").exists():
        return make_split()
    read = lambda n: set((SPLIT_DIR / n).read_text().split())
    return read("val_s1_ids.txt"), read("val_other_ids.txt")


def split_sources(side):
    """side is 'fit' (training side) or 'val'. Returns the three source frames for that side."""
    src = load_sources("train")
    val_s1, val_other = load_split()
    out = {}
    for s, df in src.items():
        ids = val_s1 if s == "source1" else val_other
        mask = df["entity_id"].isin(ids)
        out[s] = df[mask] if side == "val" else df[~mask]
    return out


def ground_truth_for(side):
    gt = load_ground_truth()
    val_s1, _ = load_split()
    return {k: v for k, v in gt.items() if (k in val_s1) == (side == "val")}


if __name__ == "__main__":
    v1, v2 = make_split()
    print(f"validation: {len(v1)} S1 entities, {len(v2)} S2/S3 records -> {SPLIT_DIR}")


def side_of(ids):
    """{entity id: 'val' or 'fit'} for training-split ids (S1 and S2/S3 alike)."""
    val_s1, val_other = load_split()
    val = val_s1 | val_other
    return {i: ("val" if i in val else "fit") for i in ids}
