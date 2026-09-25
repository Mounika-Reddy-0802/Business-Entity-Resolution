"""Abbreviation pairs learned from ground-truth matches -> data/normalised/synonyms.json.

For every matched (S1, S2/S3) pair, a token that appears on only one side is paired with a token
that appears on only the other side when the shorter one is an abbreviation of the longer one
(same first letter, letters in order: rd/road, ngr/nagar, pvt/private). Pairs seen in at least
MIN_COUNT matched records, where the long form takes at least MIN_SHARE of the short form's
pairings, become a map short -> long, separately for names and addresses. The fixed maps in
normalise.py still apply first; the learned map only adds what the data shows.

Learned from the fit side only when validating (so validation stays unseen) and from the whole
training split for test.

    python -m src.blocking.synonyms
"""
import json
from collections import Counter, defaultdict

from ..common.io_utils import DATA, load_ground_truth, load_sources
from ..common.split import load_split
from .normalise import fold

MIN_COUNT = 5
MIN_SHARE = 0.7
PATH = DATA / "normalised" / "synonyms.json"


def is_abbreviation(short, long_):
    """True if `short` is a strictly shorter in-order letter subsequence of `long_` that starts
    with the same letter (digits excluded)."""
    if len(short) >= len(long_) or not short.isalpha() or short[0] != long_[0]:
        return False
    it = iter(long_)
    return all(c in it for c in short)


def learn(pairs):
    """Map short -> long from an iterable of (text_a, text_b) of matched records."""
    counts = defaultdict(Counter)
    for a, b in pairs:
        ta, tb = set(fold(a).split()), set(fold(b).split())
        only_a, only_b = ta - tb, tb - ta
        for x in only_a:
            for y in only_b:
                if is_abbreviation(x, y):
                    counts[x][y] += 1
                elif is_abbreviation(y, x):
                    counts[y][x] += 1
    out = {}
    for short, c in counts.items():
        long_, k = c.most_common(1)[0]
        if k >= MIN_COUNT and k / sum(c.values()) >= MIN_SHARE:
            out[short] = long_
    return out


def main():
    src = load_sources("train")
    recs = {r.entity_id: r for df in src.values() for r in df.itertuples()}
    val_s1, _ = load_split()
    maps = {}
    for scope, keep in (("fit", lambda s: s not in val_s1), ("all", lambda s: True)):
        matched = [(recs[s], recs[m]) for s, ms in load_ground_truth().items() if keep(s) for m in ms]
        maps[scope] = {
            "name": learn((a.business_name, b.business_name) for a, b in matched),
            "addr": learn((a.business_address, b.business_address) for a, b in matched)}
    PATH.parent.mkdir(parents=True, exist_ok=True)
    PATH.write_text(json.dumps(maps, indent=1, sort_keys=True))
    for scope, m in maps.items():
        print(scope, {k: len(v) for k, v in m.items()},
              "e.g.", dict(list(m["addr"].items())[:8]), dict(list(m["name"].items())[:8]))


def load(scope):
    """{'name': {...}, 'addr': {...}} for scope 'fit' or 'all'; empty maps if not learned."""
    if not PATH.exists():
        return {"name": {}, "addr": {}}
    return json.loads(PATH.read_text())[scope]


if __name__ == "__main__":
    main()
