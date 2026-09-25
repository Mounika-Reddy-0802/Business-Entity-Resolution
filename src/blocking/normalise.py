"""Cleaning views per record (PLAN.md §2.1) -> data/normalised/{split}_{source}.parquet.

All views are language-agnostic: Unicode folding, punctuation removal, digit extraction and small
lookup maps. Country only selects an optional extra map; unseen countries use the generic one.

    python -m src.blocking.normalise train test
"""
import re
import sys

import pandas as pd
from unidecode import unidecode

from ..common.io_utils import DATA, SOURCES, load_sources

# canonical code for each legal-suffix spelling (after punctuation removal and letter joining)
LEGAL_SUFFIX = {
    "inc": "inc", "incorporated": "inc", "corp": "corp", "corporation": "corp", "co": "co",
    "company": "co", "cos": "co", "ltd": "ltd", "limited": "ltd", "ltda": "ltd", "llc": "llc",
    "pvt": "pvt", "private": "pvt", "pte": "pvt", "plc": "plc", "llp": "llp", "lp": "lp",
    "sarl": "sarl", "sas": "sas", "sasu": "sas", "sa": "sa", "eurl": "eurl", "sci": "sci",
    "snc": "snc", "gmbh": "gmbh", "ag": "ag", "bv": "bv", "nv": "nv", "srl": "srl", "spa": "spa",
    "pty": "pty", "opc": "opc", "pllc": "llc", "pc": "pc", "pa": "pa",
}
# suffixes that are also ordinary words are only removed at the end of the name
END_ONLY = {"co", "company", "cos", "sa", "sas", "pa", "pc", "lp", "ag", "spa", "sci"}
STOP = {"the", "of", "and", "de", "du", "la", "le", "les", "et", "des", "l", "d", "a", "au", "aux"}

# two-way address abbreviations, every spelling -> one canonical token
ADDR_MAP_GENERIC = {
    "street": "st", "str": "st", "st": "st", "road": "rd", "rd": "rd", "avenue": "ave", "av": "ave",
    "ave": "ave", "avn": "ave", "boulevard": "blvd", "blvd": "blvd", "bd": "blvd", "bvd": "blvd",
    "drive": "dr", "dr": "dr", "lane": "ln", "ln": "ln", "court": "ct", "ct": "ct", "place": "pl",
    "pl": "pl", "square": "sq", "sq": "sq", "highway": "hwy", "hwy": "hwy", "parkway": "pkwy",
    "pkwy": "pkwy", "suite": "ste", "ste": "ste", "floor": "fl", "flr": "fl", "north": "n",
    "south": "s", "east": "e", "west": "w", "near": "nr", "nr": "nr", "near by": "nr",
    "opposite": "opp", "opp": "opp", "beside": "bsd", "behind": "bhd", "market": "mkt", "mkt": "mkt",
    "nagar": "nagar", "nagr": "nagar", "ngr": "nagar", "colony": "colony", "col": "colony",
    "clny": "colony", "layout": "layout", "lyt": "layout", "sector": "sec", "sec": "sec",
    "main": "main", "cross": "crs", "crs": "crs", "chemin": "ch", "ch": "ch", "rue": "rue",
    "allee": "all", "impasse": "imp", "imp": "imp", "quai": "qu", "route": "rte", "rte": "rte",
    "saint": "st", "sainte": "ste", "mount": "mt", "mt": "mt", "building": "bldg", "bldg": "bldg",
    "apartment": "apt", "apt": "apt", "ground": "gr", "grnd": "gr", "first": "1st",
    "second": "2nd", "third": "3rd", "government": "govt", "govt": "govt", "hospital": "hosp",
    "hosp": "hosp", "station": "stn", "stn": "stn", "temple": "tmpl", "junction": "jn", "jn": "jn",
    "jct": "jn", "circle": "cir", "cir": "cir", "extension": "extn", "extn": "extn", "ext": "extn",
    "bangalore": "bengaluru", "bombay": "mumbai", "madras": "chennai", "calcutta": "kolkata",
    "poona": "pune", "gurgaon": "gurugram", "mysore": "mysuru", "cochin": "kochi",
    "trivandrum": "thiruvananthapuram", "baroda": "vadodara", "pondicherry": "puducherry",
}
# tokens that only introduce a number ("No. 12", "Shop No 4", "H.No 7", "#9")
NUMBER_WORDS = {"no", "num", "number", "shop", "h", "hno", "plot", "door", "dno", "flat", "unit"}
# per-country additions; unseen countries fall back to the generic map alone
ADDR_MAP_BY_COUNTRY = {}

# learned abbreviation maps (synonyms.py): off, 5-seed val F0.5 +0.0006 on synthetic data is noise;
# re-test on the organiser data, whose abbreviations the fixed maps may not cover
LEARNED_MAPS = False
PUNCT = re.compile(r"[^a-z0-9 ]+")
DIGITS = re.compile(r"\d+")


def fold(text):
    """Lowercase ASCII fold with & -> and, punctuation -> space, whitespace collapsed."""
    t = unidecode(str(text)).lower().replace("&", " and ").replace("'", " ")
    return " ".join(PUNCT.sub(" ", t).split())


def join_letters(tokens):
    """Merge runs of single letters: 'l l c' -> 'llc', 's a r l' -> 'sarl'."""
    out, run = [], []
    for t in tokens + [""]:
        if len(t) == 1 and t.isalpha():
            run.append(t)
            continue
        if run:
            out.append("".join(run) if len(run) > 1 else run[0])
            run = []
        if t:
            out.append(t)
    return out


def split_suffix(name_clean):
    """(name_core, legal_suffix): suffix codes removed from the end, unambiguous ones anywhere."""
    toks = join_letters(name_clean.split())
    codes = []
    while toks and (toks[-1] in LEGAL_SUFFIX or (toks[-1] == "and" and codes)):
        t = toks.pop()
        if t != "and":
            codes.append(LEGAL_SUFFIX[t])
    keep = []
    for t in toks:
        if t in LEGAL_SUFFIX and t not in END_ONLY and len(toks) > 1:
            codes.append(LEGAL_SUFFIX[t])
        else:
            keep.append(t)
    if not keep:                               # the name was only suffix words; keep it whole
        keep = toks or name_clean.split()
    return " ".join(keep), " ".join(sorted(set(codes)))


def addr_views(address, country, learned=None):
    """addr_clean, addr_numbers, postal_code, addr_tokens, city_guess for one address.
    learned: data-driven short -> long map (synonyms.py), applied before the fixed maps."""
    fixed = {**ADDR_MAP_GENERIC, **ADDR_MAP_BY_COUNTRY.get(country, {})}
    amap = {**fixed, **{k: fixed.get(v, v) for k, v in (learned or {}).items()}}
    raw = unidecode(str(address)).lower().replace("&", " and ")
    parts = [fold(p) for p in raw.split(",")]
    toks = [amap.get(t, t) for t in fold(raw).split()]
    clean = " ".join(toks)
    numbers = DIGITS.findall(clean)
    postal = max((n for n in numbers if 5 <= len(n) <= 6), key=len, default="")
    words = [t for t in toks if not t.isdigit() and t not in NUMBER_WORDS and t not in STOP]
    tail = [t for p in parts[-2:] for t in p.split() if not t.isdigit()]
    city = " ".join(amap.get(t, t) for t in tail)
    return clean, " ".join(numbers), postal, " ".join(words), city


def normalise_frame(df, learned=None):
    """Add every cleaning view to a source frame. learned: {'name': {...}, 'addr': {...}}."""
    learned = learned or {"name": {}, "addr": {}}
    out = df.copy()
    out["name_clean"] = out["business_name"].map(
        lambda n: " ".join(learned["name"].get(t, t) for t in fold(n).split()))
    core_suffix = out["name_clean"].map(split_suffix)
    out["name_core"] = [c for c, _ in core_suffix]
    out["legal_suffix"] = [s for _, s in core_suffix]
    out["name_tokens"] = out["name_core"].map(
        lambda s: " ".join(sorted(t for t in s.split() if t not in STOP) or sorted(s.split())))
    views = [addr_views(a, c, learned["addr"]) for a, c in zip(out["business_address"], out["country"])]
    for i, col in enumerate(["addr_clean", "addr_numbers", "postal_code", "addr_tokens", "city_guess"]):
        out[col] = [v[i] for v in views]
    return out


def main(splits):
    from . import synonyms                     # imported here: synonyms uses fold() from this module
    out = DATA / "normalised"
    out.mkdir(parents=True, exist_ok=True)
    for split in splits:
        for s, df in load_sources(split).items():
            learned = synonyms.load("fit" if split == "train" else "all") if LEARNED_MAPS else None
            normalise_frame(df, learned).to_parquet(out / f"{split}_{s}.parquet", index=False)
        print(split, "normalised")


def load_normalised(split):
    """{'source1': df, 'source2': df, 'source3': df} of normalised records."""
    return {s: pd.read_parquet(DATA / "normalised" / f"{split}_{s}.parquet") for s in SOURCES}


if __name__ == "__main__":
    main(sys.argv[1:] or ("train", "test"))
