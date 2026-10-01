# ML Challenge 2026: Business Entity Resolution Solution Template

**Team Name:** Gradient Descenters

**Team Members:** Shery Mounika Reddy, M Lahari, MD Rayyan, K Sai Krishna Reddy (NMIMS Hyderabad)

**Submission Date:** 27 September 2026 (final leaderboard submission); methodology submitted October 2026

---

## 1. Executive Summary

A three-stage pipeline: script-aware normalisation, name-plus-location hash blocking with a
learned candidate ranker, and a LightGBM pair classifier with competition features, followed by a
precision-first decision layer (threshold, strict one-to-one assignment). On a held-out 20% of
training entities it reaches macro F0.5 **0.9731** (public leaderboard **0.9653**) at a blocking
recall of 0.963 with 20 candidates per Source 1 entity. Everything is language-agnostic, so the unseen
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
small LightGBM *cap ranker* that keeps the 20 best candidates per entity (capped recall 0.894 →
0.963); (3) competition features over the whole candidate table, strict one-to-one assignment and
a second-stage model that compares each candidate with the entity's confident matches, exploiting
the one-to-one structure of the truth; (4) features that recognise deliberately planted
near-duplicate records (different legal form, substituted unit number, swapped name word).

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
  name: native-script and domain-style names). Key values shared by more than a block limit of
  S2/S3 records are skipped (150 for `kp`, 60 for `kn`/`knp`, 100 for `kt`/`ka`/`kaa`).
- **Join at scale:** each key kind is exploded to (key hash, record) rows, joined for all 2.2M (train)
  / 1.7M (test) S1 entities against the whole S2+S3 pool of the same country, and spilled to disk
  one kind at a time; scoring and capping run in chunks of 125k S1 entities (16 GB laptop).
- **Candidate pairs generated:** 34.47M for test (19.9 per S1 entity), 43.93M for train.
- **How true matches were not lost:** the union of keys holds 96.6% of true pairs on train (about
  245 pairs per entity before capping). A LightGBM cap ranker on cheap signals (key flags,
  skeleton/name/number similarity, empty-address flags), trained on the uncapped pairs of 20k
  fit-side entities, orders each entity's candidates; keeping the top 20 retains **96.3%** (a
  hand-weighted similarity kept only 86.9% at the same cap). Per-key recall: kp 0.55, kn 0.53,
  knp 0.62, kt 0.75, ka 0.76, kaa 0.72.
  Every key and cap was chosen from measured per-key and per-cap recall
  (`benchmarks/experiments.md`).

---

## 4. Matching Model

**Features used (96 in the submitted two-stage model):**

- Name features: Jaro-Winkler, Levenshtein ratio, token-set/sort, partial ratio on the core
  name; ratio, token-set, partial and Jaccard on the skeleton; best token-set including the DBA
  name; token Jaccard, IDF-weighted Jaccard and cosine, rarest shared token IDF, common prefix,
  first-token equality, acronym match, legal-suffix state, token counts, length ratio. (Share of
  non-Latin letters per name was tested later: 0.9656 vs 0.9655, within noise.)
- Address features: ratio, token-set/sort, partial on the clean address; token-set on the address
  skeleton; token and IDF Jaccard/cosine, number Jaccard, house-number state and containment,
  postal state, city overlap, length ratio, landmark flag, empty-address flags; name tokens found
  in the other address.
- Decoy and ambiguity features (added after error analysis; 85% of false merges were
  near-duplicate records owned by no S1 entity): legal-family conflict (Ltd vs LLP; match rate 0.4%
  when set vs 19% otherwise), substituted-vs-dropped unit numbers (9/2 vs 9/9: 2% vs 20%),
  swapped name words excluding source filler words (3% vs 43%), glued/domain names, and log
  frequency of the name and address among the country's S1 records (a record without an address
  whose name no other S1 entity has is almost surely its match).
- Other: blocking key flags; cap-ranker score; **competition features** over the whole candidate
  table (rank, gap and margin of the pair among its S1 entity's candidates; best score of the same
  S2/S3 record with any other S1 entity, margin and rank; candidate counts).

**Model type:** two-stage LightGBM. Stage 1 is a binary classifier (learning rate 0.1, 63 leaves, feature/bagging
fraction 0.8, early stopping, seed 42), 5-fold GroupKFold by S1 entity on 200k fit-side entities
(3.98M pairs), then refit on all fit rows. Stage 2 adds, from stage-1 out-of-fold scores, the
pair's rank/gap/margin within its entity and **sibling similarity**: how much the candidate
resembles the entity's three most confident matches (name skeleton, address skeleton, numbers,
raw name), since all records of one business resemble each other even when they differ from the
S1 record. Out-of-fold AUC: stage 1 0.99976, stage 2 0.99981.

**Validation:** a fixed 20% of Source 1 entities (seed 42, stratified by country and singleton)
is held out; the model never trains on them, and decision thresholds are tuned on out-of-fold
scores of the fit side and only accepted if F0.5 on the held-out entities rises (60,000 sampled
validation entities). No leaderboard feedback was used for tuning.

**Threshold selection method:** every decision rule is tuned on out-of-fold scores for macro F0.5
and kept only if it raises F0.5 on the held-out validation entities: global threshold (0.725),
per-source thresholds, relative-to-best rule, per-source caps, singleton guard. One-to-one
assignment (each S2/S3 record goes only to the S1 entity where it scores highest) is always on,
because the training truth is strictly one-to-one.

---

## 5. Results & Error Analysis

- **F_0.5 Score (macro):** 0.9731 on 60,000 held-out training entities; public leaderboard 0.9653
  (earlier versions: 0.9655 → 0.955, 0.9699 → 0.959). Seed-to-seed spread of the model is ~0.001.
- **Where the remaining score is lost (validation):** a perfect decision on our candidates would
  score 0.9866; 3.7% of true matches are never generated by blocking (native-script names at
  addresses shortened to a city, records without an address). Within the candidates the final model
  has 953 false merges and 5,274 missed true candidates (from 1,138 and 6,426 before the decoy and
  ambiguity features).
- **Common false positives (wrong merges):** 85% are planted near-duplicates owned by no S1 entity:
  the same business with a different legal form ("Rn Brothers Ltd" vs "Rn Brothers LLP"), a
  neighbouring unit (Unit 9/2 vs 9/9, Flat 104 vs 125), or one name word replaced ("Gdb Logistics"
  vs "Gdb Power").
- **Common false negatives (missed matches):** candidates with an empty address whose name is shared
  by several businesses; gibberish trade names at an identical address; native-script or
  domain-style names whose address also changed.

| step | val F0.5 |
|---|---|
| name-pair blocking (recall 0.894), 64 features, threshold 0.70, one-to-one | 0.9574 |
| + address-pair key and learned cap ranker (recall 0.960) | 0.9655 (LB 0.955) |
| + looser block limits (recall 0.963) + stage-2 sibling model | 0.9699 (LB 0.959) |
| + decoy and ambiguity features (final) | 0.9731 (LB 0.9653) |

---

## 6. Conclusion

Most of the score comes from candidate generation that respects how this data repeats names
across businesses (name + location keys) and from exploiting the one-to-one structure (competition
features, one-to-one assignment). Script-aware normalisation carries the Indian records, and the
same language-agnostic code handles France, whose predicted matches per entity (3.30) and
no-match share (5.8%) mirror the training distribution (3.46, 5.6%). The main lesson: the largest
gains came from error analysis (which records were missed or wrongly merged, and why), not from
tuning; the next step would be similarity search over whole records to recover the 3.7% of matches
that blocking never generates.

---

## Appendix

### A. Code Artefacts

`code/business_entity_resolution/`: `src/common` (I/O, split, metric, input and submission
checks, release packaging), `src/blocking` (normalise, block), `src/matching` (features,
train_lgbm, decide, sanity, errors, cross_country, tune, stability), `src/neural` (optional
embeddings, off). Entry point: `bash scripts/run_pipeline.sh` with the organiser data in
`data/raw/dataset/`; it writes `output/matching_results.tsv` and `output/candidate_pairs.tsv` and
runs the organiser validator. Models: LightGBM (MIT) only; no external data, lookups, APIs or
pretrained models. Libraries: pandas, pyarrow, numpy, scikit-learn, rapidfuzz, jellyfish,
Unidecode, LightGBM (versions pinned in `requirements.txt`).

### B. Additional Results

Per-key recall, per-cap recall, every tried change and its validation score, with commit hashes:
`benchmarks/experiments.md` and `benchmarks/raw/*.json`.
