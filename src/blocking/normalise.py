"""Cleaning views per record (PLAN.md §2.1) -> data/normalised/{split}_{source}.parquet.

All views are language-agnostic: transliteration to ASCII (Devanagari, Tamil, Telugu, ... via
unidecode), placeholder and bracket junk removed, punctuation stripped, digits extracted, small
lookup maps, and a consonant skeleton that makes transliterated and Latin spellings meet
("rAm mArkeTing prAiveT" and "Ram Marketing Private" -> "rm mrktng prvt"). Country only selects
an optional extra map; unseen countries use the generic one.

Records are processed in parallel chunks, so the 5M-row source files take minutes, not hours.

    python -m src.blocking.normalise train test
"""
import os
import re
import sys
from multiprocessing import Pool

import pandas as pd
from unidecode import unidecode

from ..common.io_utils import DATA, RAW, SOURCES, is_fresh, load_tsv

# canonical code for each legal-suffix spelling (after punctuation removal and letter joining)
LEGAL_SUFFIX = {
    "inc": "inc", "incorporated": "inc", "corp": "corp", "corporation": "corp", "co": "co",
    "company": "co", "cos": "co", "ltd": "ltd", "limited": "ltd", "ltda": "ltd", "llc": "llc",
    "pvt": "pvt", "private": "pvt", "pte": "pvt", "plc": "plc", "llp": "llp", "lp": "lp",
    "sarl": "sarl", "sas": "sas", "sasu": "sas", "sa": "sa", "eurl": "eurl", "sci": "sci",
    "snc": "snc", "gmbh": "gmbh", "ag": "ag", "bv": "bv", "nv": "nv", "srl": "srl", "spa": "spa",
    "pty": "pty", "opc": "opc", "pllc": "llc", "pc": "pc", "pa": "pa",
    # transliterated Indian-script spellings seen in the data (unidecode of the native words)
    "praaivett": "pvt", "praiveett": "pvt", "limittedd": "ltd", "limitedd": "ltd",
    "elelpi": "llp", "elelpii": "llp", "elelsi": "llc",
}
# suffixes that are also ordinary words are only removed at the start or end of the name
EDGE_ONLY = {"co", "company", "cos", "sa", "sas", "pa", "pc", "lp", "ag", "spa", "sci"}
STOP = {"the", "of", "and", "de", "du", "la", "le", "les", "et", "des", "l", "d", "a", "au", "aux"}
# tokens that stand for a missing value
PLACEHOLDER = {"null", "none", "nan", "na", "n a", "nil", "unknown", "not available"}

# two-way address abbreviations, every spelling -> one canonical token
ADDR_MAP_GENERIC = {
    "street": "st", "str": "st", "st": "st", "road": "rd", "rd": "rd", "avenue": "ave", "av": "ave",
    "ave": "ave", "avn": "ave", "boulevard": "blvd", "blvd": "blvd", "bd": "blvd", "bvd": "blvd",
    "drive": "dr", "dr": "dr", "lane": "ln", "ln": "ln", "court": "ct", "ct": "ct", "place": "pl",
    "pl": "pl", "square": "sq", "sq": "sq", "highway": "hwy", "hwy": "hwy", "parkway": "pkwy",
    "pkwy": "pkwy", "suite": "ste", "ste": "ste", "floor": "fl", "flr": "fl", "north": "n",
    "south": "s", "east": "e", "west": "w", "near": "nr", "nr": "nr",
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
    "terrace": "ter", "ter": "ter", "trail": "trl", "trl": "trl", "township": "twp", "twp": "twp",
    "bangalore": "bengaluru", "bombay": "mumbai", "madras": "chennai", "calcutta": "kolkata",
    "poona": "pune", "gurgaon": "gurugram", "mysore": "mysuru", "cochin": "kochi",
    "trivandrum": "thiruvananthapuram", "baroda": "vadodara", "pondicherry": "puducherry",
}
# tokens that only introduce a number ("No. 12", "Shop No 4", "H.No 7", "#9")
NUMBER_WORDS = {"no", "num", "number", "shop", "h", "hno", "plot", "door", "dno", "flat", "unit",
                "ph", "box", "po"}
# per-country additions; unseen countries fall back to the generic map alone
ADDR_MAP_BY_COUNTRY = {}

PUNCT = re.compile(r"[^a-z0-9 ]+")
DIGITS = re.compile(r"\d+")
JUNK = re.compile(r"<\s*null\s*>|\[+|\]+|<<|>>|\{|\}")
DOMAIN = re.compile(r"\b(?:www\.)?([a-z0-9-]+)\.(?:com|net|org|in|co\.in|fr|us|biz|info)\b")
DBA = re.compile(r"\b(?:d\s*\.?\s*b\s*\.?\s*a\s*\.?|doing business as|trading as|t/a)\s+")
PHONE = re.compile(r"\d[\d\s-]{6,}\d")
VOWELS = re.compile(r"[aeiouy]")
REPEAT = re.compile(r"(.)\1+")
ASPIRATE = re.compile(r"([bcdgjkpt])h")
SKEL_MAP = str.maketrans("cqwzx", "kkvjk")
CHUNK = 250_000


def fold(text):
    """Lowercase ASCII with junk removed: & -> and, placeholders and brackets dropped,
    punctuation -> space, whitespace collapsed."""
    t = unidecode(str(text)).lower()
    t = JUNK.sub(" ", t).replace("&", " and ").replace("+", " and ").replace("'", " ")
    t = DOMAIN.sub(r" \1 ", t)
    toks = PUNCT.sub(" ", t).split()
    return " ".join(x for x in toks if x not in PLACEHOLDER)


def skeleton(text):
    """Consonant skeleton of each token: ph->f, aspirated h dropped, c/q/x->k, w->v, z->j,
    vowels removed, repeated letters collapsed. Digits are kept."""
    out = []
    for w in text.split():
        s = ASPIRATE.sub(r"\1", w.replace("ph", "f")).translate(SKEL_MAP)
        s = REPEAT.sub(r"\1", VOWELS.sub("", s)) or w[:1]
        out.append(s)
    return " ".join(out)


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
    """(name_core, legal_suffix): suffix codes removed anywhere (the data moves them to the front,
    "LLC Crystal Staffing"), ambiguous ones only at the start or end."""
    toks = join_letters(name_clean.split())
    codes, keep = [], []
    last = len(toks) - 1
    for i, t in enumerate(toks):
        code = LEGAL_SUFFIX.get(t)
        if code and len(toks) > 1 and (t not in EDGE_ONLY or i in (0, last)):
            codes.append(code)
        else:
            keep.append(t)
    while keep and keep[-1] == "and" and codes:     # "prem and co" -> "prem"
        keep.pop()
    if not keep:                                    # the name was only suffix words; keep it whole
        keep = toks or name_clean.split()
    return " ".join(keep), " ".join(sorted(set(codes)))


def name_views(name):
    """name_clean, name_core, legal_suffix, name_tokens, name_skel, name_alt for one name."""
    raw = unidecode(str(name)).lower()
    raw = PHONE.sub(" ", raw)
    alt = ""
    m = DBA.search(raw)
    if m:
        alt = fold(raw[m.end():])
        raw = raw[:m.start()] + " " + raw[m.end():]
    clean = fold(raw)
    core, suffix = split_suffix(clean)
    tokens = " ".join(sorted(t for t in core.split() if t not in STOP) or sorted(core.split()))
    return clean, core, suffix, tokens, skeleton(tokens), alt


def addr_views(address, country):
    """addr_clean, addr_numbers, postal_code, addr_tokens, city_guess, addr_skel for one address."""
    amap = {**ADDR_MAP_GENERIC, **ADDR_MAP_BY_COUNTRY.get(country, {})}
    raw = unidecode(str(address)).lower()
    parts = [fold(p) for p in raw.split(",")]
    parts = [p for p in parts if p]
    toks = [amap.get(t, t) for p in parts for t in p.split()]
    clean = " ".join(toks)
    numbers = [n.lstrip("0") or "0" for n in DIGITS.findall(clean)]
    postal = max((n for n in DIGITS.findall(clean) if 5 <= len(n) <= 6), key=len, default="")
    words = [t for t in toks if not t.isdigit() and t not in NUMBER_WORDS and t not in STOP]
    tail = [amap.get(t, t) for p in parts[-2:] for t in p.split() if not t.isdigit()]
    return (clean, " ".join(numbers), postal, " ".join(words), " ".join(tail),
            skeleton(" ".join(sorted(set(words)))))


NAME_COLS = ["name_clean", "name_core", "legal_suffix", "name_tokens", "name_skel", "name_alt"]
ADDR_COLS = ["addr_clean", "addr_numbers", "postal_code", "addr_tokens", "city_guess", "addr_skel"]


def normalise_frame(df):
    """Add every cleaning view to a source frame."""
    out = df.copy()
    names = [name_views(n) for n in out["business_name"]]
    addrs = [addr_views(a, c) for a, c in zip(out["business_address"], out["country"])]
    for i, col in enumerate(NAME_COLS):
        out[col] = [v[i] for v in names]
    for i, col in enumerate(ADDR_COLS):
        out[col] = [v[i] for v in addrs]
    return out


def normalise_parallel(df, workers=None):
    """normalise_frame over row chunks on all cores."""
    chunks = [df.iloc[i:i + CHUNK] for i in range(0, len(df), CHUNK)]
    if len(chunks) == 1:
        return normalise_frame(df)
    with Pool(workers or os.cpu_count()) as pool:
        return pd.concat(pool.map(normalise_frame, chunks), ignore_index=True)


def main(splits):
    out = DATA / "normalised"
    out.mkdir(parents=True, exist_ok=True)
    for split in splits:
        outputs = [out / f"{split}_{s}.parquet" for s in SOURCES]
        inputs = [RAW / split / f"{split}_{s}.tsv" for s in SOURCES]
        if is_fresh(outputs, inputs, ["blocking/normalise.py"]):
            print(split, "normalised (cached)")
            continue
        for s in SOURCES:
            df = load_tsv(RAW / split / f"{split}_{s}.tsv")
            normalise_parallel(df).to_parquet(out / f"{split}_{s}.parquet", index=False)
            print(split, s, len(df), "normalised", flush=True)


def load_normalised(split, columns=None):
    """{'source1': df, 'source2': df, 'source3': df} of normalised records."""
    return {s: pd.read_parquet(DATA / "normalised" / f"{split}_{s}.parquet", columns=columns)
            for s in SOURCES}


if __name__ == "__main__":
    main(sys.argv[1:] or ("train", "test"))
