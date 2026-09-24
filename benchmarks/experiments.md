# Experiment log

Every run that changes a number gets a row. `raw/` holds the json each row came from.

`commit` is HEAD when the run started; a trailing `+` means the run included the uncommitted
changes that land in the commit adding the row. `(synthetic)` rows ran on the stand-in dataset
(docs/problems.md) and are not comparable with real-data rows.

| time (IST) | who | commit | change | block recall | cands/S1 | val F0.5 | val F0.5 US | val F0.5 IN | cross-country F0.5 | uploaded? | LB |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 01:08 | krishna-2-005 | 2285f9a+ | naive exact-name blocker, 4 rapidfuzz features, lgbm, t=0.5 (synthetic) | 0.5397 | 1.3660 | 0.7231 | 0.7454 | 0.7007 |  | no |  |
| 01:13 | krishna-2-005 | c776e2f+ | normalise views + keys k1-k6, cap 40 by best per-field cosine rank (synthetic) | 0.9992 | 39.9770 |  |  |  |  | no |  |
| 01:18 | krishna-2-005 | 50b2c6c+ | 71 features incl. competition, lgbm 5-fold groupkfold, t=0.5 (synthetic) | 0.9992 | 39.9770 | 0.9876 | 0.9900 | 0.9852 |  | no |  |
