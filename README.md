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

Outputs: `output/matching_results.tsv` (upload this) and `output/candidate_pairs.tsv`. Tested on
a 16 GB, 20-thread Windows laptop; a full run from scratch takes about 5 hours (blocking ~3 h,
features ~45 min, training ~20 min). Stages cache their outputs, so reruns skip unchanged work
(`FORCE=1` recomputes).

| stage | module | output |
|---|---|---|
| input checks (columns, id prefixes, duplicates, truth ids) | `src/common/check_inputs.py` | stdout |
| fixed 20% validation split of S1 (seed 42), stratified by country and singleton | `src/common/split.py` | `data/splits/` |
| normalisation: transliteration, skeleton, junk/suffix clean-up | `src/blocking/normalise.py` | `data/normalised/` |
| blocking: name+location hash keys, learned cap ranker, top 20 per S1 | `src/blocking/block.py` | `data/candidates/` |
| 65+ pair features incl. competition features, per country, in parts | `src/matching/features.py` | `data/features/` |
| LightGBM, 5-fold GroupKFold by S1 entity on a 200k-entity sample | `src/matching/train_lgbm.py` | `models/`, `data/scores/` |
| decision rules tuned on OOF, accepted on validation; one-to-one | `src/matching/decide.py` | `models/decision.json`, `output/` |
| organiser validator, local checks, per-country sanity report | `utils/`, `src/common/check_submission.py`, `src/matching/sanity.py` | `benchmarks/raw/` |

Tools: `python -m src.matching.errors` (worst validation entities), `python -m
src.matching.stability` (seed spread), `python -m src.common.release archive <n> --lb <score>`
(after each upload), `python -m src.common.release package <team>` (final zip), `pytest -q tests`.

## Results (validation: 60,000 held-out training entities)

Every number is a row in `benchmarks/experiments.md` with its json in `benchmarks/raw/`.

| version | blocking recall @20 | val F0.5 | US | India |
|---|---|---|---|---|
| name-pair keys + hand-scored cap (v2) | 0.869 | — | — | — |
| + learned cap ranker, model, threshold 0.70, one-to-one (sub 1) | 0.941 | 0.9574 | 0.9710 | 0.9374 |
| + address-pair key (sub 2) | 0.960 | **0.9655** | 0.9757 | 0.9506 |

Models: LightGBM (MIT). No external data, lookups or pretrained models are used. An optional
multilingual embedding stage (`src/neural/embeddings.py`, Apache-2.0 MiniLM) exists but is off.
