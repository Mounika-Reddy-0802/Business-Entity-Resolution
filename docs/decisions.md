# Decisions

One line per decision: time IST, who, what was decided, why.

- 2026-09-25 — team — file contracts frozen (docs/contracts.md); a change needs an entry here and a message in the group.
- 2026-09-25 01:05 — krishna-2-005 — build and test every stage on a synthetic dataset with the exact schema until the organiser files are downloaded; numbers from it are labelled synthetic and not compared with the leaderboard.
- 2026-09-25 01:05 — krishna-2-005 — run_pipeline.sh uses the organiser validator when utils/validate_submission.py exists and src/common/check_submission.py otherwise; the organiser script is never written or edited by us.
- 2026-09-25 01:16 — krishna-2-005 — block per side inside train (fit S1 x fit S2/S3, val S1 x val S2/S3) so validation blocking sees the same kind of pool as test.
- 2026-09-25 01:16 — krishna-2-005 — cap ranks candidates by the better of their name-cosine and address-cosine ranks (recall at 20 rises from 0.960 to 0.998 vs name+0.5*address); cap 40.
- 2026-09-25 01:16 — krishna-2-005 — keep K1-K5 although leave-one-out recall shows only K6 matters on synthetic data: they are cheap, feed key-flag features and the real data may need them; re-check on the organiser data.
- 2026-09-25 01:16 — krishna-2-005 — K2 blocks on the first significant token, the first sorted token and the rarest token (key values with more than 200 S2/S3 records are skipped).
- 2026-09-25 01:24 — krishna-2-005 — harder synthetic v2 (20% chain branches in 4 cities, 6% city-only addresses, same-name same-city distractors); rows before this one used v1 and are not comparable.
- 2026-09-25 01:26 — lahari-66 — decision rules are tuned on fit-side OOF scores and accepted only if validation F0.5 rises, so validation is never used to pick a threshold value directly.
- 2026-09-25 01:31 — lahari-66 — a feature group is dropped only if removing it lifts both cross-country directions by >= 0.002; only is_s3 qualified (+0.0029/+0.0030), and val F0.5 rose 0.9799 -> 0.9809 without it. Dropped list and params live in code (train_lgbm.py) so a clean clone reproduces them.
- 2026-09-25 01:40 — lahari-66 — one-typo legal-suffix detection reverted: val 0.9809 -> 0.9801 on synthetic (singletons +0.003, matched -0.003); re-test on the organiser data where suffix typos may be more common.
- 2026-09-25 01:58 — Mounika-Reddy-0802 — optional embeddings use sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2: licence Apache-2.0, 117.7M parameters, downloaded from the Hugging Face Hub into models/ (gitignored); inference only, no fine-tuning, no external data.
- 2026-09-25 02:15 — Mounika-Reddy-0802 — embeddings disabled (ENABLED = False): K7 + cosine features 0.9811 and cosine features alone 0.9813 vs 0.9838 without; synthetic variants are character-level, so re-test on the organiser data before the final package.
- 2026-09-25 05:45 — Rayyan-Mohammed — seed noise on val F0.5 is std 0.0012 (5 seeds, range 0.9811-0.9842; seed 42 is the best of them): from now on a change is kept only if its 5-seed mean beats 0.9831 by more than ~2 std (0.0024). Earlier single-seed gains below that (learning rate, threshold/one-to-one) are within noise; they stay because they are harmless, not because they were shown to help.
- 2026-09-25 06:10 — Rayyan-Mohammed — abbreviation maps learned from matched training pairs (synonyms.py) are built but off (LEARNED_MAPS = False): 5-seed mean 0.9837 vs 0.9831 is within noise on synthetic data, whose abbreviations the fixed maps already cover; first thing to re-test on the organiser data.
- 2026-09-25 10:05 — Rayyan-Mohammed — learning rate back to 0.05: 5-seed mean 0.9829 vs 0.9831 at 0.02 (noise), and a full run takes ~6 min instead of ~22; faster iteration matters more on the real data.
