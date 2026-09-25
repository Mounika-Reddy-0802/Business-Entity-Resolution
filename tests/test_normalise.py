from src.blocking.normalise import addr_views, fold, split_suffix


def test_fold_handles_accents_ampersand_and_punctuation():
    assert fold("Épicerie L'Étoile & Fils, S.A.R.L.") == "epicerie l etoile and fils s a r l"


def test_suffix_split_keeps_core_and_canonical_codes():
    assert split_suffix(fold("Ganesh Agencies Private Limited")) == ("ganesh agencies", "ltd pvt")
    assert split_suffix(fold("Prairie Tax Services, L.L.C.")) == ("prairie tax services", "llc")
    assert split_suffix(fold("Boulangerie Dupont S.A.R.L.")) == ("boulangerie dupont", "sarl")
    assert split_suffix(fold("Kaveri Kirana Store & Co")) == ("kaveri kirana store", "co")


def test_suffix_only_name_is_kept_whole():
    core, _ = split_suffix(fold("Limited"))
    assert core == "limited"


def test_address_views_abbreviations_postal_and_city_alias():
    clean, numbers, postal, tokens, city = addr_views(
        "Shop No 12, Gandhi Nagr, Opposite SBI ATM, Bangalore - 560018", "India")
    assert "nagar" in clean.split() and "opp" in clean.split()
    assert postal == "560018" and numbers.split() == ["12", "560018"]
    assert "bengaluru" in city.split()
    assert "shop" not in tokens.split()


def test_unseen_country_uses_generic_map():
    clean, _, postal, _, _ = addr_views("12 av. Victor Hugo, 75011 Paris", "Neverland")
    assert clean == "12 ave victor hugo 75011 paris" and postal == "75011"


def test_loader_keeps_quotes_and_drops_bom(tmp_path):
    from src.common.io_utils import load_tsv
    path = tmp_path / "x.tsv"
    path.write_bytes('﻿entity_id\tbusiness_name\nS1-1\t"Joe\'s" Diner\nS1-2\tAcme "Best\n'.encode("utf-8"))
    df = load_tsv(path)
    assert list(df.columns) == ["entity_id", "business_name"]
    assert df.business_name.tolist() == ['"Joe\'s" Diner', 'Acme "Best']


def test_abbreviation_learning_needs_support():
    from src.blocking.synonyms import MIN_COUNT, is_abbreviation, learn
    assert is_abbreviation("rd", "road") and is_abbreviation("ngr", "nagar")
    assert not is_abbreviation("road", "rd") and not is_abbreviation("dr", "road")
    pairs = [("12 Main Rd", "12 Main Road")] * MIN_COUNT + [("Oak Ln", "Oak Lane")] * (MIN_COUNT - 1)
    assert learn(pairs) == {"rd": "road"}
