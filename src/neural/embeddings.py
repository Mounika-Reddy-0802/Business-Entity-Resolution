"""Sentence embeddings of the normalised name and address -> data/embeddings/{split}_{source}.npz.

Model: sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2 (Apache-2.0, 117.7M params),
inference only (docs/decisions.md). Used by blocking key K7 and by the embedding-cosine features.
Set ENABLED = False to run the pipeline without it.

    python -m src.neural.embeddings train test
"""
if __name__ == "__main__":
    import torch  # noqa: F401  must load before pyarrow on Windows (docs/problems.md)

import hashlib
import os
import sys

import numpy as np

from ..blocking.normalise import load_normalised
from ..common.io_utils import DATA, ROOT, SOURCES

ENABLED = False           # off: lowered val F0.5 on synthetic data (benchmarks/experiments.md)
MODEL = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
OUT = DATA / "embeddings"


def encode(texts, model):
    """L2-normalised float32 embeddings; empty strings get a zero vector."""
    texts = list(texts)
    emb = model.encode([t or " " for t in texts], batch_size=256, normalize_embeddings=True,
                       convert_to_numpy=True, show_progress_bar=False).astype(np.float32)
    emb[[i for i, t in enumerate(texts) if not t]] = 0.0
    return emb


def text_key(df):
    """Hash of the embedded texts, so a cache is reused only for identical inputs."""
    return hashlib.sha1("\n".join(df.name_core + "\t" + df.addr_clean).encode()).hexdigest()


def main(splits):
    if not ENABLED:
        print("embeddings disabled")
        return
    os.environ.setdefault("HF_HOME", str(ROOT / "models" / "hf"))
    import torch
    from sentence_transformers import SentenceTransformer
    torch.manual_seed(42)
    model = SentenceTransformer(MODEL, device="cpu")
    OUT.mkdir(parents=True, exist_ok=True)
    for split in splits:
        for s, df in load_normalised(split).items():
            path = OUT / f"{split}_{s}.npz"
            key = text_key(df)
            if path.exists() and str(np.load(path, allow_pickle=True)["key"]) == key:
                continue
            np.savez(path, ids=df.entity_id.to_numpy(), key=key, name=encode(df.name_core, model),
                     addr=encode(df.addr_clean, model))
        print(split, "embedded")


def load(split):
    """{'ids', 'name', 'addr'} arrays over all sources of a split, or None when disabled/missing."""
    if not ENABLED or not all((OUT / f"{split}_{s}.npz").exists() for s in SOURCES):
        return None
    parts = [np.load(OUT / f"{split}_{s}.npz", allow_pickle=True) for s in SOURCES]
    ids = np.concatenate([p["ids"] for p in parts])
    return {"ids": ids, "name": np.vstack([p["name"] for p in parts]),
            "addr": np.vstack([p["addr"] for p in parts])}


if __name__ == "__main__":
    main(sys.argv[1:] or ("train", "test"))
