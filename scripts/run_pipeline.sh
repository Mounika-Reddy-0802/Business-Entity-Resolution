#!/usr/bin/env bash
# End to end: split -> normalise -> block -> features -> train -> decide -> validate.
# Decision rules are tuned on the fit side (validation held out); then, unless FULL=0, the model
# is retrained on the whole training split before scoring test.
set -e
cd "$(dirname "$0")/.."
if [ -x .venv/bin/python ]; then PY=.venv/bin/python
elif [ -x .venv/Scripts/python.exe ]; then PY=.venv/Scripts/python.exe
else PY="${PYTHON:-python3}"; fi
$PY -m src.common.split
$PY -m src.blocking.normalise train test
$PY -m src.neural.embeddings train test
$PY -m src.blocking.block train test
$PY -m src.matching.features train test
$PY -m src.matching.train_lgbm
$PY -m src.matching.train_lgbm --predict
$PY -m src.matching.decide sweep
if [ "${FULL:-1}" = 1 ]; then
  $PY -m src.matching.train_lgbm --full
  $PY -m src.matching.train_lgbm --predict
fi
$PY -m src.matching.decide test
if [ -f utils/validate_submission.py ]; then
  $PY utils/validate_submission.py --matching output/matching_results.tsv \
    --candidate output/candidate_pairs.tsv --test-dir data/raw/dataset/test
fi
$PY -m src.common.check_submission --matching output/matching_results.tsv \
  --candidate output/candidate_pairs.tsv --test-dir data/raw/dataset/test
$PY -m src.matching.sanity
