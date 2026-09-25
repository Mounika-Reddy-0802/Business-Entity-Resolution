"""Shared I/O helpers. Every stage reads and writes through these so the three lanes agree on
paths, column names and TSV handling. Do not change signatures without a team decision
(docs/decisions.md)."""
import csv
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
RAW = ROOT / "data" / "raw" / "dataset"
DATA = ROOT / "data"
OUTPUT = ROOT / "output"

SOURCES = ("source1", "source2", "source3")


def data_is_synthetic():
    """True while data/raw/dataset holds the synthetic stand-in, not the organiser files."""
    return (RAW / "SYNTHETIC").exists()


def load_tsv(path):
    """Tab-separated, everything as string, empty cells stay empty strings (never NaN).
    Quotes are ordinary characters: the files are unquoted, and a name starting with `"` must keep
    it. A UTF-8 byte-order mark, if present, is dropped from the first header."""
    return pd.read_csv(path, sep="\t", dtype=str, keep_default_na=False, quoting=csv.QUOTE_NONE,
                       encoding="utf-8-sig")


def load_sources(split):
    """split is 'train' or 'test'. Returns {'source1': df, 'source2': df, 'source3': df}."""
    return {s: load_tsv(RAW / split / f"{split}_{s}.tsv") for s in SOURCES}


def parse_id_list(cell):
    return [x.strip() for x in cell.split(",") if x.strip()] if cell else []


def load_ground_truth():
    """{s1_id: [matched ids]} for the whole training set (empty list for singletons)."""
    df = load_tsv(RAW / "train" / "train_ground_truth.tsv")
    return {r.source1_entity_id: parse_id_list(r.matched_entity_ids) for r in df.itertuples()}


def read_id_list_tsv(path, column):
    df = load_tsv(path)
    return {r.source1_entity_id: parse_id_list(getattr(r, column)) for r in df.itertuples()}


def pairs_to_map(df, id_col="cand_id"):
    """DataFrame with columns s1_id and id_col -> {s1_id: [ids]}."""
    return df.groupby("s1_id")[id_col].apply(list).to_dict()


def is_fresh(outputs, inputs, code=None):
    """True when every output exists and is newer than every input and every code file the stage
    depends on (module paths relative to src/, e.g. "blocking/normalise.py"; default: all of
    src/), so a deterministic stage can be skipped. FORCE=1 in the environment always reruns."""
    import os
    outputs, inputs = [Path(p) for p in outputs], [Path(p) for p in inputs]
    if os.environ.get("FORCE") == "1" or not all(p.exists() for p in outputs + inputs):
        return False
    src = ROOT / "src"
    code_files = [src / c for c in code] if code else list(src.rglob("*.py"))
    newest_in = max(p.stat().st_mtime for p in inputs + code_files)
    return min(p.stat().st_mtime for p in outputs) > newest_in


def write_pairs_tsv(pairs, s1_ids, path, column, id_col="cand_id"):
    """Vectorised writer for large outputs: one row per S1 id in s1_ids order, its ids from the
    (s1_id, id_col) pair frame joined with commas (empty when it has none). Pairs must be unique."""
    lists = pairs.groupby("s1_id", sort=False)[id_col].agg(",".join)
    out = pd.DataFrame({"source1_entity_id": list(s1_ids)})
    out[column] = out.source1_entity_id.map(lists).fillna("")
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(path, sep="\t", index=False)
