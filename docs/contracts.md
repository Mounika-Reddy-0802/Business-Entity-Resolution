# File contracts

Every stage reads and writes exactly these files and columns. Changing one needs a line in
docs/decisions.md.

| Stage | Module | Writes | Columns |
|---|---|---|---|
| split | src/common/split.py | data/splits/val_s1_ids.txt, val_other_ids.txt | one id per line |
| normalise | src/blocking/normalise.py | data/normalised/{split}_{source}.parquet | entity_id, business_name, business_address, country, name_clean, name_core, legal_suffix, name_tokens, addr_clean, addr_numbers, postal_code, addr_tokens, city_guess |
| block | src/blocking/block.py | data/candidates/{split}_candidates.parquet | s1_id, cand_id, k1..k7 (bool), ngram_name_cos, ngram_addr_cos, emb_cos (optional) |
| features | src/matching/features.py | data/features/{split}_features.parquet | s1_id, cand_id, label (train only), then numeric feature columns |
| train | src/matching/train_lgbm.py | models/lgbm.txt, data/scores/train_oof.parquet | s1_id, cand_id, p |
| predict | src/matching/train_lgbm.py --predict | data/scores/test_scores.parquet | s1_id, cand_id, p |
| decide | src/matching/decide.py | output/matching_results.tsv, output/candidate_pairs.tsv, models/decision.json | official format |
| evaluate | src/common/evaluate.py | benchmarks/raw/<timestamp>_<tag>.json | metrics for experiments.md |

`{split}` is `train` or `test`. Inside `train`, the validation side is selected by the id files
from the split stage; never by a separate file.
