# File contracts

Every stage reads and writes exactly these files and columns. Changing one needs a line in
docs/decisions.md.

| Stage | Module | Writes | Columns |
|---|---|---|---|
| input check | src/common/check_inputs.py | (stdout only) | — |
| split | src/common/split.py | data/splits/val_s1_ids.txt | one S1 id per line |
| normalise | src/blocking/normalise.py | data/normalised/{split}_{source}.parquet | entity_id, business_name, business_address, country, name_clean, name_core, legal_suffix, name_tokens, name_skel, name_alt, addr_clean, addr_numbers, postal_code, addr_tokens, city_guess, addr_skel |
| block | src/blocking/block.py | data/candidates/{split}_candidates.parquet | s1_id, cand_id, kp, kr, ka, kz (bool), name_sim, addr_sim, cheap_score |
| features | src/matching/features.py | data/features/{split}/part_NNNN.parquet | s1_id, cand_id, candidate columns, competition columns, string features; train adds side (fit/val) and label |
| train | src/matching/train_lgbm.py | models/lgbm.txt, models/train_metrics.json, data/scores/train_oof.parquet | s1_id, cand_id, p, label |
| predict | src/matching/train_lgbm.py --predict | data/scores/val_scores.parquet, data/scores/test_scores.parquet | s1_id, cand_id, p |
| decide | src/matching/decide.py | models/decision.json, output/matching_results.tsv, output/candidate_pairs.tsv | official format |
| evaluate | src/common/evaluate.py | benchmarks/raw/<timestamp>_<tag>.json | metrics for experiments.md |

`{split}` is `train` or `test`. Blocking and competition features always use the whole split. In
`train`, only a fixed sample of S1 entities (features.SAMPLE_FIT from the fit side, SAMPLE_VAL
from the validation side of data/splits) is featurised; the validation side is never used to fit.
