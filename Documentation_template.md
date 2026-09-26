# ML Challenge 2026: Business Entity Resolution Solution Template

**Team Name:** [Your Team Name]
**Team Members:** Krishna (krishna-2-005), Lahari (lahari-66), Mounika (Mounika-Reddy-0802), Rayyan (Rayyan-Mohammed)
**Submission Date:** 27 September 2026

---

## 1. Executive Summary

A three-stage pipeline: script-aware normalisation, name-plus-location hash blocking with a
learned candidate ranker, and a LightGBM pair classifier with competition features, followed by a
precision-first decision layer (threshold, strict one-to-one assignment). On a held-out 20% of
training entities it reaches macro F0.5 **0.9699** at a blocking recall of
0.96 with 20 candidates per Source 1 entity. Everything is language-agnostic, so the unseen
country (France) runs through the same code.

---

## 2. Methodology

### 2.1 Problem Analysis

- **Scale:** train 2.21M S1 / 5.03M S2 / 5.29M S3 records; test 1.73M / 4.89M / 5.08M.
- **Label structure:** 5.6% of S1 entities are singletons; the rest have 3.46 matches on average
  (up to 11). 74% of S2/S3 records are matched and **no S2/S3 record belongs to two S1 entities**,
  so the truth is strictly one-to-one.
- **Names repeat across businesses:** 86 different US businesses are called "Blue Hypnosis"; a
  name alone does not identify an entity, name plus location does.
- **Scripts:** ~23% of Indian S2/S3 names are in Devanagari, Tamil, Telugu, Kannada, Gujarati,
  Bengali, Punjabi or Malayalam ("ராஜ் இன்வெஸ்ட்மெண்ட்ஸ் எல்எல்பி" = "Raj Investments LLP");
  state names too ("தமிழ்நாடு", "महाराष्ट्र").
- **Noise:** typos, word swaps, legal suffixes moved to the front ("LLC Crystal Staffing"),
  bracket junk (`[[LLC]]`, `>>`), placeholders (`null`, `<NULL>`), domain-style names
  (`maurewilliamscolombier.com`), DBA names, phone numbers, reordered and truncated addresses,
  3.4% empty addresses, US states as codes or names.

### 2.2 Solution Strategy

**Approach Type:** Blocking + learned candidate ranker + gradient-boosted pair classifier +
decision layer.
**Core Innovation:** (1) a consonant *skeleton* view that makes transliterated and Latin spellings
meet; (2) name-plus-location blocking keys joined as 64-bit hashes over the full data, with a
small LightGBM *cap ranker* that keeps the 20 best candidates per entity (recall 0.941 → 0.960 with
the address-pair key); (3) competition features computed over the whole candidate table, and
strict one-to-one assignment, exploiting the one-to-one structure of the truth.

---

## 3. Candidate Generation (Blocking)

Normalisation (`src/blocking/normalise.py`): Unicode transliteration to ASCII, junk and
placeholder removal, `&`/`+` → `and`, legal suffixes removed wherever they occur and kept as a
canonical code, DBA alternative name, two-way address abbreviation map, digits with leading zeros
dropped, and a **consonant skeleton** per token (ph→f, aspirated h dropped, c/q/x→k, w→v, vowels
removed, repeats collapsed): "Raj Investments" and "ராஜ் இன்வெஸ்ட்மெண்ட்ஸ்" both become
`nvstmnts rj`.

- **Blocking keys used** (all combined with the country as an equality constraint and hashed):
  `kp` pair of name skeleton tokens; `kn` full name skeleton + one address word; `knp` name-token
  pair + one address number; `kt` one name token + one address word (survives a typo in the
  others); `ka` address number + address word; `kaa` pair of address words (same address, any
  name: native-script and domain-style names). Key values shared by more than 60 (150 for `kp`)
  S2/S3 records are skipped.
- **Candidate pairs generated:** 34.35M for test (19.8 per S1 entity), 43.8M for train.
- **How true matches were not lost:** the union of keys holds 96.1% of true pairs on train. A
  LightGBM cap ranker on cheap signals (key flags, skeleton/number similarity, empty-address
  flags), trained on the uncapped pairs of 20k fit-side entities, orders each entity's candidates;
  keeping the top 20 retains 95.95% (a hand-weighted similarity kept only 86.9% at the same cap).
  Every key and cap was chosen from measured per-key and per-cap recall
  (`benchmarks/experiments.md`).

---

## 4. Matching Model

**Features used (65 in the submitted model):**
- Name features: Jaro-Winkler, Levenshtein ratio, token-set/sort, partial ratio on the core
  name; ratio, token-set, partial and Jaccard on the skeleton; best token-set including the DBA
  name; token Jaccard, IDF-weighted Jaccard and cosine, rarest shared token IDF, common prefix,
  first-token equality, acronym match, legal-suffix state, token counts, length ratio. (Share of
  non-Latin letters per name was tested later: 0.9656 vs 0.9655, within noise.)
- Address features: ratio, token-set/sort, partial on the clean address; token-set on the address
  skeleton; token and IDF Jaccard/cosine, number Jaccard, house-number state and containment,
  postal state, city overlap, length ratio, landmark flag, empty-address flags; name tokens found
  in the other address.
- Other: blocking key flags; cap-ranker score; **competition features** over the whole candidate
  table (rank, gap and margin of the pair among its S1 entity's candidates; best score of the same
  S2/S3 record with any other S1 entity, margin and rank; candidate counts).

**Model type:** two-stage LightGBM. Stage 1 is a binary classifier (learning rate 0.1, 63 leaves, feature/bagging
fraction 0.8, early stopping, seed 42), 5-fold GroupKFold by S1 entity on 200k fit-side entities
(3.98M pairs), then refit on all fit rows. Stage 2 adds, from stage-1 out-of-fold scores, the
pair's rank/gap/margin within its entity and **sibling similarity**: how much the candidate
resembles the entity's three most confident matches (name skeleton, address skeleton, numbers,
raw name), since all records of one business resemble each other even when they differ from the
S1 record. Stage 2 lifts OOF AUC from 0.99961 to 0.99972.
**Threshold selection method:** every decision rule is tuned on out-of-fold scores for macro F0.5
and kept only if it raises F0.5 on the held-out validation entities: global threshold (0.725),
per-source thresholds, relative-to-best rule, per-source caps, singleton guard. One-to-one
assignment (each S2/S3 record goes only to the S1 entity where it scores highest) is always on,
because the training truth is strictly one-to-one.

---

## 5. Results & Error Analysis

- **F_0.5 Score (macro):** 0.9699 on 60,000 held-out training entities (leaderboard for the
  previous 0.9655 version: 0.955). Seed-to-seed spread of the model is ~0.001.
- **Common false positives (wrong merges):** neighbouring businesses with near-identical names on
  the same street ("Gauthier Culture", 498 Town Line Rd vs "Gauthier Couture", 519 Town Line Rd);
  records with an empty address whose name is shared by several entities.
- **Common false negatives (missed matches):** candidates never generated (two-thirds of misses):
  native-script names at addresses shortened to a city, and records with no address; model misses:
  native-script or domain-style names whose address also changed.

| step | val F0.5 |
|---|---|
| name-pair blocking (recall 0.894), 64 features, threshold 0.70, one-to-one | 0.9574 |
| + address-pair key and learned cap ranker (recall 0.960) | 0.9655 (LB 0.955) |
| + looser block limits (recall 0.963) + stage-2 sibling model | 0.9699 |

---

## 6. Conclusion

Most of the score comes from candidate generation that respects how this data repeats names
across businesses (name + location keys) and from exploiting the one-to-one structure (competition
features, one-to-one assignment). Script-aware normalisation carries the Indian records, and the
same language-agnostic code handles France, whose predicted matches per entity (3.41) and
singleton share (5%) mirror the training distribution.

---

## Appendix

### A. Code Artefacts

`code/business_entity_resolution/`: `src/common` (I/O, split, metric, input and submission
checks, release packaging), `src/blocking` (normalise, block), `src/matching` (features,
train_lgbm, decide, sanity, errors, cross_country, tune, stability), `src/neural` (optional
embeddings, off). Entry point: `bash scripts/run_pipeline.sh` with the organiser data in
`data/raw/dataset/`; it writes `output/matching_results.tsv` and `output/candidate_pairs.tsv` and
runs the organiser validator. Models: LightGBM (MIT) only; no external data or services.

### B. Additional Results

Per-key recall, per-cap recall, every tried change and its validation score, with commit hashes:
`benchmarks/experiments.md` and `benchmarks/raw/*.json`.
