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

## 2026-09-25 02:58 IST — run_pipeline.sh broke on a Python path with spaces
- Symptom: clean-clone run with `PYTHON=".../Business Entity Resolution/.venv/Scripts/python.exe"`
  stopped at the first stage (`/c/Users/HP/Downloads/Business: No such file or directory`).
- Fix: quote `"$PY"` on every call. The clean clone then reproduced val F0.5 0.9842 and
  byte-identical `output/*.tsv`.
- Cost: 5 minutes.

## 2026-09-26 07:24 IST — disk full during test features (page file growth)
- Symptom: `OSError: [Errno 28] No space left on device` while writing test feature parts; free
  space on C: fell from 17 GB to 3 GB within minutes and came back when the process was stopped.
- Cause: the per-row country lookup `cands.s1_id.map(...)` turned 34M Arrow strings into Python
  objects; the process spilled to the Windows page file, which grew until the disk was full (the
  drive is ~97% used by other data).
- Fix: country codes via Arrow's C++ `index_in` (integer codes, no Python strings); a resource
  logger (`data/resources.log`) records free disk and RAM every 5 minutes during long runs.
- Cost: ~40 minutes and one restart of the v4 chain.
