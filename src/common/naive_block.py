"""Day-0 blocker so the pipeline runs end to end before the real one exists.
Key = country + lowercase alphanumeric name with common legal suffixes removed.
Writes data/candidates/{split}_candidates.parquet with columns s1_id, cand_id, k_naive."""
import re
import sys
from collections import defaultdict

import pandas as pd
from unidecode import unidecode

from .io_utils import DATA, load_sources

SUFFIX = {"inc", "incorporated", "corp", "corporation", "co", "company", "ltd", "limited", "llc",
          "pvt", "private", "plc", "llp", "sarl", "sas", "sa", "eurl", "gmbh", "the"}


def name_key(name):
    toks = re.sub(r"[^a-z0-9 ]", " ", unidecode(name).lower().replace("&", " and ")).split()
    return " ".join(t for t in toks if t not in SUFFIX)


def block(split):
    src = load_sources(split)
    index = defaultdict(list)
    for s in ("source2", "source3"):
        for r in src[s].itertuples():
            index[(r.country, name_key(r.business_name))].append(r.entity_id)
    rows = [(r.entity_id, c)
            for r in src["source1"].itertuples()
            for c in index.get((r.country, name_key(r.business_name)), [])]
    df = pd.DataFrame(rows, columns=["s1_id", "cand_id"]).drop_duplicates()
    df["k_naive"] = True
    out = DATA / "candidates"
    out.mkdir(parents=True, exist_ok=True)
    df.to_parquet(out / f"{split}_candidates.parquet", index=False)
    return df


if __name__ == "__main__":
    for split in sys.argv[1:] or ("train", "test"):
        print(split, len(block(split)), "candidate pairs")
