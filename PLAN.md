# PLAN.md — Business Entity Resolution, Amazon ML Challenge 2026

One laptop, sequential work, three days (25–27 Sep 2026, IST). Goal: a top private-leaderboard
finish on macro F0.5. This file is the plan of record; `docs/decisions.md` records deviations.

---

## 1. What wins this challenge

The metric is macro F0.5 per Source 1 entity, singletons included. That means:

1. **Blocking recall is the ceiling.** A true match that never becomes a candidate is lost forever.
   Target ≥ 0.98 recall on validation with ≤ ~25 candidates per S1 entity.
2. **Precision is worth twice recall.** Decision thresholds are chosen by maximising F0.5 on
   validation, never F1. Expect the operating threshold well above 0.5.
3. **Singletons are free points.** An S1 entity with no true match scores 1.0 for an empty list
   and 0.0 for any prediction. A separate, higher "singleton guard" threshold is a distinct knob.
4. **Source 1 is deduplicated**, so one S2/S3 record almost never belongs to two S1 entities.
   One-to-one assignment (Hungarian or greedy by probability) and "competition" features
   (how well this candidate scores for *other* S1 entities) are the biggest single gains after
   blocking.
5. **France is only in test.** Every feature must be language-agnostic (character n-grams,
   digits, token sets after Unicode folding). Country is used as an equality/blocking constraint
   only, never as a filter or one-hot. A cross-country validation (train US → validate India and
   the reverse) is the proxy for France; features that do not transfer are dropped.
6. **The public leaderboard is a subset.** Validation on a large held-out split is trusted over
   small public-LB moves. Uploads are used to confirm, not to search.

---

## 2. Pipeline

```
raw TSVs
  -> src/common/split.py           fixed 80/20 split (seed 42), stratified by country and singleton
  -> src/blocking/normalise.py     cleaning views per record          -> data/normalised/
  -> src/blocking/block.py         union of blocking keys K1..K7       -> data/candidates/
  -> src/matching/features.py      pairwise features                   -> data/features/
  -> src/matching/train_lgbm.py    LightGBM, grouped 5-fold OOF        -> data/scores/, models/
  -> src/matching/decide.py        thresholds, 1-to-1, singleton guard -> output/*.tsv
  -> utils/validate_submission.py  PASS before any upload
```
Exact files and columns: `docs/contracts.md`. Every stage caches to parquet so reruns skip
unchanged stages.

### 2.1 Normalisation views
- `name_clean`: lower, Unicode NFKD fold (é→e), `&`→`and`, punctuation stripped, whitespace
  collapsed.
- `name_core`: `name_clean` minus legal suffixes; the suffix is kept separately as
  `legal_suffix` (INC, LTD, PVT, LLC, CORP, CO, PLC, LLP, SARL, SAS, SA, EURL, GMBH, …).
- `name_tokens`: sorted tokens of `name_core` minus stop-words (the, of, and, de, du, la, le,
  les, et, des).
- `addr_clean`: same folding plus a two-way abbreviation map (st/street, rd/road, ave/avenue,
  blvd, nr/near, opp/opposite, mkt/market, nagar, colony, rue, av, bd, chemin, place, …).
- `addr_numbers`: all digit groups; `postal_code`: longest 5–6-digit group.
- `addr_tokens`: tokens of `addr_clean` minus numbers; `city_guess`: last one or two tokens.
- Abbreviation and suffix lists are extended from token frequency tables of the training data
  (top tokens per country), not from memory alone.

### 2.2 Blocking keys (all constrained to equal country)
| Key | Definition | Catches |
|---|---|---|
| K1 | `name_core` exact | suffix-only variation |
| K2 | first significant token of `name_core` | truncated / DBA names |
| K3 | first 6 chars of the sorted-token string | word reordering |
| K4 | Double-Metaphone of the first two tokens | typos, transliteration |
| K5 | `postal_code`; and house number + first street token | same address, different name |
| K6 | TF-IDF char 3-gram cosine top-20 on `name_core`, and separately on `addr_clean` | everything fuzzy |
| K7 | sentence-embedding top-20 (optional, licensed model ≤ 8B) | cross-script / semantic |

Union, dedupe, cap at 60 per S1 entity by K6 score. Per-key recall is measured and keys that add
candidates without recall are dropped. `candidate_pairs.tsv` is exactly this final set.

### 2.3 Pairwise features (~50, all numeric, language-agnostic)
- Name: Jaro-Winkler, Levenshtein ratio, token-set, token-sort, partial ratio, token Jaccard,
  char 3-gram TF-IDF cosine, word TF-IDF cosine, common-prefix length, first-token equality,
  IDF-weighted shared rare tokens.
- Legal suffix: equal / one missing / conflict.
- Address: the same string metrics on `addr_clean`; Jaccard of `addr_tokens` and of
  `addr_numbers`; house-number equality; postal equal / one missing / conflict; city equality;
  length ratio; landmark keyword flag.
- Cross-field: name tokens found in the other record's address and vice versa; combined
  name+address char-gram cosine.
- Competition/context: rank of the candidate among the S1 entity's candidates by name cosine and
  by address cosine; gap to the best candidate; candidate count for the S1 entity; the
  candidate's best score against *any other* S1 entity and how many S1 entities list it.
- Source and country: candidate source (S2/S3), country equality, token counts.
- Optional: embedding cosine for name, address, concatenation.

### 2.4 Model
LightGBM binary classifier, positives = ground-truth pairs inside the candidate set, negatives =
all other candidate pairs. 5-fold CV grouped by S1 entity gives out-of-fold probabilities for
threshold tuning. `scale_pos_weight` and `num_leaves`/`min_child_samples` tuned on validation
F0.5 after the decision layer, not on AUC alone. Feature importance is saved to
`benchmarks/raw/`. Optional challenger: a fine-tuned multilingual cross-encoder (MIT/Apache,
< 1B) whose score is fed back into LightGBM as a feature; kept only if validation F0.5 improves.

### 2.5 Decision layer (tuned on validation F0.5, in this order)
1. Global threshold `t` (sweep 0.30–0.95).
2. Per-source thresholds `t_s2`, `t_s3`.
3. Relative rule: keep only `p ≥ alpha × p_best` for the entity (alpha 0.6–0.8).
4. One-to-one assignment of each S2/S3 id to at most one S1 entity (Hungarian on the score
   matrix, greedy fallback).
5. Cardinality caps learned from the training distribution of matches per entity (per source).
6. Singleton guard `t_single`: if `p_best < t_single`, output an empty list.
7. Sanity: only candidate ids, no S1 ids, no duplicates; validator runs at the end of `decide.py`.
All chosen values are written to `models/decision.json` and to `benchmarks/experiments.md`.

---

## 3. Schedule

### Session 1 — tonight, unattended, ~7 hours (Day 1)
Executed by the prompt in `PROMPT.md`. Deliverables at the end: a validated
`output/matching_results.tsv` ready to upload, blocking recall ≥ 0.97, LightGBM + tuned decision
layer, cross-country check, `docs/session_2026-09-25_1.md` with the numbers and the next three
steps, everything pushed to `main`.

| Elapsed | Work | Done when |
|---|---|---|
| 0:00–0:20 | Repo setup, env, dataset check, hook, tests, EDA facts to `docs/eda.md` | tests pass, EDA committed |
| 0:20–0:50 | Split, naive blocker, end-to-end with a trivial matcher, validator PASS | first row in experiments.md |
| 0:50–1:50 | Normalisation + K1–K6 blocking, per-key recall report | recall ≥ 0.97 or best achievable, logged |
| 1:50–3:00 | Features + LightGBM grouped CV, OOF scores | OOF AUC and val F0.5 with plain threshold logged |
| 3:00–4:00 | Decision layer sweeps 1–7 | each rule's delta logged; best config in decision.json |
| 4:00–4:40 | Cross-country validation; drop non-transferring features | cross-country F0.5 logged |
| 4:40–5:40 | Error analysis loop (missed matches → blocking; false merges → features) | each iteration logged |
| 5:40–6:20 | Optional embeddings K7 + cosine features (skip if install/download > 10 min) | delta logged or "skipped" |
| 6:20–7:00 | Retrain on full train, regenerate test outputs, validator PASS, README, template draft, session note | pushed to main |
| after 7:00 | Keep iterating on 4:40–5:40 style improvements until stopped | every gain logged |

### Day 1 morning (after waking)
- Read `docs/session_2026-09-25_1.md` and `benchmarks/experiments.md`.
- Upload `output/matching_results.tsv` (submission 1). Compare LB with validation F0.5; if they
  differ by more than ~3 points, investigate the split before anything else.
- Tag `sub-1`, save the file to `benchmarks/raw/submission_1.tsv.gz`.
- Spend remaining uploads (max 5/day) one hypothesis at a time; always keep one in reserve.

### Day 2 (26 Sep)
- Morning: error analysis on validation with the LB-confirmed model; extend blocking dictionaries
  from the top-token tables; add features that separate the observed false merges.
- Afternoon: challenger model if a GPU or enough CPU time exists; stack or blend only if
  validation improves; retune the decision layer after every model change.
- Evening: freeze normalisation and blocking; 5-seed CV for stability; fill
  `Documentation_template.md` from experiments.md.

### Day 3 (27 Sep)
- Morning: retrain the final config on the full training set; regenerate outputs on a clean clone
  using only the README; diff against the previous outputs and explain every difference.
- Midday: final upload plus one safer variant (slightly higher `t_single`). Do not chase the
  public LB on the last day.
- Afternoon: `pip freeze` into requirements.txt, README walkthrough, template review, build
  `<team>_submission.zip` with `output/`, `code/business_entity_resolution/{src,README.md,requirements.txt}`,
  `Documentation_template.md`. Tag `v1.0`. Upload by 22:00 IST; keep the last slot unused.

---

## 4. Risks
| Risk | Mitigation |
|---|---|
| France scores badly | cross-country validation, Unicode folding, no country filters, French suffix/abbreviation entries added to the maps |
| Blocking misses | union of keys, per-key recall, char-gram ANN fallback, cap tuned against recall |
| Over-merging on singletons | singleton guard, 1-to-1 assignment, cardinality caps, predicted-count distribution compared with training |
| Validation vs LB disagree | grouped split, leakage check (no S2/S3 id on both sides), per-country F0.5 |
| Licence / size violation | model name, licence, params logged in decisions.md before use |
| Format rejection | validator wired into decide.py; never upload without PASS |
| Cannot reproduce | pinned requirements, seeds, single `scripts/run_pipeline.sh`, clean-clone run on Day 3 |

---

## 5. Checklists
**Before every upload:** validator PASS · row count = test S1 count · every matched id is a
candidate for the same entity · predicted match-count distribution resembles training · per-country
predicted counts sane including France · experiments.md row with commit hash.

**Final package:** matching_results.tsv identical to the last upload · candidate_pairs.tsv is the
exact scored set · src complete, README reproduces end to end · requirements pinned, model
licences listed · Documentation_template.md complete · no external data anywhere · zip structure
exact · `v1.0` tagged.
