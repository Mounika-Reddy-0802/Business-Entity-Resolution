"""Fixed validation split of Source 1 entities. data/ is gitignored, so every machine regenerates it
with the same seed and gets the identical split.

Validation entities are 20% of Source 1, stratified by country and by singleton/non-singleton.
Source 2/3 records are not split: blocking runs over the whole training pool, as it does on test,
and the model is fitted on fit-side entities only (see features.train_sample).

    python -m src.common.split
"""
import random
from collections import defaultdict

import pandas as pd

from .io_utils import DATA, RAW, load_ground_truth

SEED = 42
VAL_FRACTION = 0.2
SPLIT_DIR = DATA / "splits"


def make_split(seed=SEED, val_fraction=VAL_FRACTION):
    """Write data/splits/val_s1_ids.txt (only when its content changes) and return the id set."""
    s1 = pd.read_csv(RAW / "train" / "train_source1.tsv", sep="\t", dtype=str, keep_default_na=False,
                     quoting=3, usecols=["entity_id", "country"])
    gt = load_ground_truth()
    strata = defaultdict(list)
    for eid, country in zip(s1.entity_id, s1.country):
        strata[(country, len(gt.get(eid, [])) == 0)].append(eid)
    rng = random.Random(seed)
    val_s1 = set()
    for key in sorted(strata):
        ids = sorted(strata[key])
        rng.shuffle(ids)
        val_s1.update(ids[: int(round(len(ids) * val_fraction))])
    SPLIT_DIR.mkdir(parents=True, exist_ok=True)
    path, text = SPLIT_DIR / "val_s1_ids.txt", "\n".join(sorted(val_s1)) + "\n"
    if not path.exists() or path.read_text() != text:        # unchanged file keeps its mtime
        path.write_text(text)
    return val_s1


def load_split():
    """Set of validation S1 ids (the split is made on first use)."""
    path = SPLIT_DIR / "val_s1_ids.txt"
    return set(path.read_text().split()) if path.exists() else make_split()


def ground_truth_for(side):
    """Ground truth restricted to the 'fit' or 'val' S1 entities."""
    gt = load_ground_truth()
    val_s1 = load_split()
    return {k: v for k, v in gt.items() if (k in val_s1) == (side == "val")}


if __name__ == "__main__":
    print(f"validation: {len(make_split())} S1 entities -> {SPLIT_DIR}")
