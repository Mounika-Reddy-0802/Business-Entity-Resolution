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
