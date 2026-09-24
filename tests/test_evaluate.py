from src.common.evaluate import blocking_recall, f05_entity, macro_f05


def test_worked_example_from_statement():
    assert abs(f05_entity(["S2-00047", "S2-00193", "S3-00812"], ["S2-00047", "S3-00812"]) - 0.714) < 0.001


def test_singleton_rule():
    assert f05_entity([], []) == 1.0
    assert f05_entity(["S2-1"], []) == 0.0
    assert f05_entity([], ["S2-1"]) == 0.0


def test_macro_and_recall():
    true = {"a": ["S2-1"], "b": [], "c": ["S3-9", "S2-2"]}
    pred = {"a": ["S2-1"], "c": ["S3-9"]}
    expected = (1 + 1 + (1.25 * 1 * 0.5) / (0.25 * 1 + 0.5)) / 3
    assert abs(macro_f05(pred, true) - expected) < 1e-9
    assert abs(blocking_recall({"a": ["S2-1", "S2-5"], "c": ["S2-2"]}, true) - 2 / 3) < 1e-9
