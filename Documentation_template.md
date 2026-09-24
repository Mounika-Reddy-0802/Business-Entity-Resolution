# Business Entity Resolution — Methodology

Team: Krishna · Lahari · Mounika · Rayyan. Amazon ML Challenge 2026.

> Status: draft written after session 1. The organiser template was not in the package on the
> development machine, so this file uses the headings the problem statement asks for. All
> numbers so far come from a synthetic stand-in dataset with the organiser schema
> (docs/problems.md) and will be replaced by real-data numbers.

## 1. Methodology

The task is framed as candidate generation followed by pairwise classification and a
set-level decision layer:

1. A fixed 80/20 validation split of Source 1 entities (seed 42), stratified by country and by
   singleton / matched. S2/S3 records matched to a validation entity stay on the validation side,
   so blocking, features and scoring on validation see a pool shaped like the test pool.
2. Normalisation views per record, blocking with a union of keys, capped candidate lists.
3. About 80 numeric, language-agnostic pair features, including competition features that
   compare a pair with the other candidates of its S1 entity and with the other S1 entities that
   list the same S2/S3 record.
4. LightGBM in two stages with 5-fold GroupKFold by S1 entity: stage 2 adds context computed from
   stage-1 out-of-fold scores.
5. A decision layer tuned for macro F0.5 on out-of-fold scores, where each rule is accepted only
   if it raises validation F0.5.
6. Retrain on the full training split and write the two submission files.

Country is treated as an open set of strings: it only constrains blocking (equal country) and
selects optional per-country dictionaries, which fall back to generic maps for unseen labels
(France). No external data, lookups or APIs are used.

## 2. Candidate generation / blocking

Normalisation (`src/blocking/normalise.py`): Unicode folding (é→e), `&`→`and`, punctuation
removal, single-letter runs joined (`l.l.c.`→`llc`), legal suffixes split off into a canonical
code (`private limited`→`pvt ltd`, `s.a.r.l.`→`sarl`), stop-words removed for the sorted-token
view; addresses get a two-way abbreviation map (street/st, nagar/nagr, near/nr, avenue/av, ...),
digit groups, a postal code (longest 5–6 digit group) and a city guess from the last components.

Keys, all within equal country:

| key | definition |
|---|---|
| K1 | exact `name_core` |
| K2 | first significant token, first sorted token, and rarest token of the name |
| K3 | first 6 characters of the sorted-token string |
| K4 | metaphone of the first two tokens |
| K5 | postal code; house number + first street word |
| K6 | TF-IDF char 3-gram cosine top-20 on name and, separately, on address (chunked sparse products) |
| K7 | sentence-embedding top-20 (optional, off) |

Key values shared by more than 200 S2/S3 records are skipped. The union is capped at 40
candidates per S1 entity, ranked by the better of the name-cosine rank and the address-cosine
rank, so a candidate that is strong on either field alone survives. On synthetic data this gives
recall 0.997 at 40 candidates per entity (0.960 at a cap of 20 with a name-weighted ranking vs
0.998 with the rank rule).

## 3. Model architecture and feature engineering

Feature groups (`src/matching/features.py`):

- Name: Jaro-Winkler, Levenshtein ratio, token-set, token-sort, partial ratio, token Jaccard,
  IDF-weighted Jaccard, IDF of the rarest shared token, char 3-gram and word TF-IDF cosine,
  common prefix, first-token equality, acronym match, token counts and length ratio.
- Legal suffix: both empty / equal / one missing / conflict.
- Address: the same string metrics on the clean address, token and number Jaccard, house-number
  and postal-code state, postal prefix equality, city overlap, length ratio, landmark flag.
- Cross-field: name tokens found in the other record's address and the reverse; char cosine of
  name+address.
- Competition: for four similarity scores, rank / gap / margin among the S1 entity's candidates,
  and the best score the candidate reaches with any other S1 entity, with the margin and rank.
- Key flags from blocking.

Model (`src/matching/train_lgbm.py`): LightGBM binary classifier (learning rate 0.02, 63 leaves,
feature and bagging fraction 0.8, seed 42, deterministic), early stopping inside 5-fold
GroupKFold by S1 entity, final model with 1.1x the mean best round. Stage 2 repeats this with
nine extra features from stage-1 OOF scores (rank, gap and margin within the S1 entity, sum of
scores, best score of the same record for another S1 entity). `is_s3` is dropped (cross-country
check below).

## 4. Decision layer

Tuned on fit-side OOF scores for macro F0.5, in this order, and kept only if validation F0.5 rises:
global threshold, per-source thresholds, relative rule `p >= alpha * p_best`, one-to-one
assignment of each S2/S3 record to its best S1 entity, per-source cardinality caps from the
training distribution, singleton guard. Current config: per-source thresholds S2 0.40 / S3 0.45,
other rules off (stage 2 already learns one-to-one from the context features).

## 5. Transfer to an unseen country

France appears only in test. Proxy: train on one training country, score the other. India→US
0.9735 and US→India 0.9663 vs about 0.982 in-country. Feature-group ablation showed no group
collapsing transfer; removing `is_s3` improved both directions by 0.003 and was dropped.
Predicted matches per entity on test are close to the training distribution in every country,
France included (`src/matching/sanity.py`).

## 6. Experiments

| change | val F0.5 |
|---|---|
| naive blocker baseline (synthetic v1) | 0.723 |
| K1–K6 blocking + 71 features + LightGBM, t=0.5 | 0.9791 |
| + threshold 0.40, one-to-one | 0.9799 |
| + drop `is_s3` | 0.9809 |
| typo-tolerant suffixes (reverted) | 0.9801 |
| + stage-2 context model | 0.9838 |
| multilingual MiniLM embeddings, K7 + cosines (not kept) | 0.9811 |
| embedding cosines only (not kept) | 0.9813 |
| learning rate 0.02, best of 14 one-change LightGBM variants | 0.9842 |

Full log with commits: `benchmarks/experiments.md`.

## 7. Models and licences

- LightGBM (MIT) trained from scratch on the training data.
- Optional, currently disabled: sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2
  (Apache-2.0, 117.7M parameters), inference only.
