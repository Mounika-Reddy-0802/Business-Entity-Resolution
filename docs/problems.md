# Problems

Add an entry the moment you lose time to something: symptom, cause, fix, what it cost.

## 2026-09-25 01:05 IST — organiser dataset and validator not on this machine
- Symptom: `data/raw/dataset/{train,test}` and `utils/validate_submission.py` absent; searched the
  home directory, Downloads and every zip under 600 MB for `train_source1.tsv`,
  `train_ground_truth.tsv` and `validate_submission.py` — nothing found. Only the problem
  statement and guidelines PDFs are in Downloads.
- Cause: the files are distributed through the challenge portal and were not downloaded here.
- Workaround: `scripts/make_synthetic_data.py` writes a synthetic dataset with the exact schema
  (train: US + India, 10k S1; test: adds France, 5k S1) and the noise patterns from the problem
  statement. `src/common/check_submission.py` applies every format rule from the statement and
  stands in for the validator. Every number logged in this session is on synthetic data and
  says so; the pipeline is built so the real files drop into `data/raw/dataset/` and
  `utils/` and `bash scripts/run_pipeline.sh` reruns end to end.
- Cost: all tuned values (thresholds, caps, hyperparameters) must be re-tuned on the real data.

## 2026-09-25 01:24 IST — rapidfuzz 3.9.6 segfaults in multithreaded cpdist on Windows
- Symptom: `process.cpdist(..., workers=-1)` on ~200k pairs crashes with an access violation;
  `workers=1` works.
- Fix: pin rapidfuzz 3.14.1, which runs the same call multithreaded without crashing.
- Cost: 5 minutes.

## 2026-09-25 02:08 IST — torch DLL init fails when pyarrow is loaded first (Windows)
- Symptom: `OSError: [WinError 1114] ... c10.dll` when `src.neural.embeddings` imports
  sentence-transformers after pandas/pyarrow; `import pyarrow; import torch` reproduces it.
- Fix: the embeddings stage imports torch before anything else when run as a module.
- Cost: 5 minutes.
