import pandas as pd

from src.common.evaluate import macro_f05
from src.common.io_utils import pairs_to_map
from src.matching.decide import BASE, apply, fast_f05, truth_frame

SCORES = pd.DataFrame({
    "s1_id": ["S1-1", "S1-1", "S1-1", "S1-2", "S1-2", "S1-3"],
    "cand_id": ["S2-1", "S3-1", "S2-2", "S2-2", "S2-3", "S3-9"],
    "p": [0.95, 0.70, 0.40, 0.90, 0.30, 0.55],
})


def picked(cfg):
    return sorted(zip(*apply(SCORES, {**BASE, **cfg})[["s1_id", "cand_id"]].values.T))


def test_per_source_thresholds():
    assert picked({"t_s2": 0.5, "t_s3": 0.8}) == [("S1-1", "S2-1"), ("S1-2", "S2-2")]


def test_relative_rule_and_singleton_guard():
    assert ("S1-1", "S3-1") not in picked({"t_s2": 0.3, "t_s3": 0.3, "alpha": 0.8})
    assert all(s != "S1-3" for s, _ in picked({"t_s2": 0.3, "t_s3": 0.3, "t_single": 0.6}))


def test_one_to_one_gives_a_record_to_its_best_entity_only():
    got = picked({"t_s2": 0.3, "t_s3": 0.3, "one_to_one": True})
    assert ("S1-2", "S2-2") in got and ("S1-1", "S2-2") not in got


def test_caps_keep_the_highest_scores():
    got = picked({"t_s2": 0.3, "t_s3": 0.3, "cap_s2": 1})
    assert [c for s, c in got if s == "S1-1" and c.startswith("S2-")] == ["S2-1"]


def test_fast_f05_matches_reference_metric():
    truth = {"S1-1": ["S2-1", "S3-1"], "S1-2": [], "S1-3": ["S3-9"], "S1-4": []}
    sel = apply(SCORES, {**BASE, "t_s2": 0.5, "t_s3": 0.5})
    assert abs(fast_f05(sel, truth_frame(truth), list(truth)) - macro_f05(pairs_to_map(sel), truth)) < 1e-12


def test_second_largest_per_group():
    from src.matching.ranking import second_largest
    got = second_largest([0.2, 0.9, 0.5, 0.7, 0.1], ["a", "a", "a", "b", "c"])
    assert list(got) == [0.5, 0.5, 0.5, 0.0, 0.0]


def test_write_pairs_tsv_keeps_order_and_empty_rows(tmp_path):
    from src.common.io_utils import read_id_list_tsv, write_pairs_tsv
    pairs = pd.DataFrame({"s1_id": ["S1-2", "S1-1", "S1-2"], "cand_id": ["S2-9", "S3-1", "S3-4"]})
    path = tmp_path / "m.tsv"
    write_pairs_tsv(pairs, ["S1-1", "S1-2", "S1-3"], path, "matched_entity_ids")
    assert path.read_text().splitlines()[0] == "source1_entity_id\tmatched_entity_ids"
    assert read_id_list_tsv(path, "matched_entity_ids") == {"S1-1": ["S3-1"], "S1-2": ["S2-9", "S3-4"], "S1-3": []}
