# Business Entity Resolution — Amazon ML Challenge 2026

Match Source 2 / Source 3 business records to the deduplicated Source 1 reference using only the
provided name, address and country fields. Scored on macro F0.5 per Source 1 entity.

## Reproduce end to end

```bash
python -m venv .venv && source .venv/bin/activate      # Windows: .venv\Scripts\activate
pip install -r requirements.txt
# organiser files: data/raw/dataset/{train,test}/*.tsv and utils/validate_submission.py
bash scripts/run_pipeline.sh
```

`run_pipeline.sh` runs, in order:

| stage | module | output |
|---|---|---|
| fixed 80/20 split (seed 42), stratified by country and singleton | `src/common/split.py` | `data/splits/` |
| normalisation views (folding, legal suffixes, address abbreviations, digits) | `src/blocking/normalise.py` | `data/normalised/` |
| optional sentence embeddings (off by default) | `src/neural/embeddings.py` | `data/embeddings/` |
| blocking: keys K1–K6 (+K7), union, cap 40 per S1 entity | `src/blocking/block.py` | `data/candidates/` |
| ~80 pairwise features incl. competition features | `src/matching/features.py` | `data/features/` |
| LightGBM, 5-fold GroupKFold by S1 entity, two stages | `src/matching/train_lgbm.py` | `models/`, `data/scores/` |
| decision rules tuned on OOF, accepted on validation | `src/matching/decide.py sweep` | `models/decision.json` |
| retrain on the whole training split (`FULL=0` skips) | `src/matching/train_lgbm.py --full` | `models/` |
| test outputs | `src/matching/decide.py test` | `output/matching_results.tsv`, `output/candidate_pairs.tsv` |
| format checks and pre-upload sanity report | `utils/validate_submission.py`, `src/common/check_submission.py`, `src/matching/sanity.py` | `benchmarks/raw/` |

A full run takes about 3.5 minutes on 10k training S1 entities (about 10 minutes at 4x that size).
Release: `python -m src.common.release archive <n> [--lb <score>]` after each upload
(keeps `benchmarks/raw/submission_<n>.tsv.gz`), `python -m src.common.release package <team>` for
the final zip; both refuse synthetic data and rerun the validators.
Tools: `python -m src.common.eda` (dataset facts to `docs/eda.md`),
`python -m src.matching.cross_country` (train one country, score the other),
`python -m src.matching.errors` (worst validation entities), `pytest -q tests`.

## Data status

The organiser dataset was not on the development machine during session 1 (docs/problems.md).
Every stage was built and measured on a synthetic stand-in with the exact schema
(`python scripts/make_synthetic_data.py`; it writes `data/raw/dataset/SYNTHETIC`). **The numbers
below are synthetic and say nothing about leaderboard performance.** To switch to the real data:
delete `data/`, put the organiser files in place, rerun `bash scripts/run_pipeline.sh`.

## Results (validation side, synthetic data v2)

Every number is a row in `benchmarks/experiments.md` with its json in `benchmarks/raw/`.

| change | block recall | cands/S1 | val F0.5 |
|---|---|---|---|
| naive exact-name blocker, 4 fuzzy features (synthetic v1) | 0.540 | 1.4 | 0.723 |
| K1–K6 blocking, 71 features, LightGBM, t=0.5 | 0.997 | 40 | 0.9791 |
| + tuned threshold 0.40 and one-to-one assignment | 0.997 | 40 | 0.9799 |
| + drop `is_s3` after cross-country check | 0.997 | 40 | 0.9809 |
| + stage-2 model on stage-1 score context | 0.997 | 40 | 0.9838 |
| + learning rate 0.02 (best of 14 one-change variants, current) | 0.997 | 40 | **0.9842** |

Seed noise: over 5 LightGBM seeds the current configuration scores 0.9831 ± 0.0012 (range
0.9811–0.9842; seed 42 is the best), so single-seed differences below ~0.0025 are not evidence.

Cross-country (train one country, score the other): India→US 0.9735, US→India 0.9663.
