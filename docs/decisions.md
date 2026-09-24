# Decisions

One line per decision: time IST, who, what was decided, why.

- 2026-09-25 — team — file contracts frozen (docs/contracts.md); a change needs an entry here and a message in the group.
- 2026-09-25 01:05 — krishna-2-005 — build and test every stage on a synthetic dataset with the exact schema until the organiser files are downloaded; numbers from it are labelled synthetic and not compared with the leaderboard.
- 2026-09-25 01:05 — krishna-2-005 — run_pipeline.sh uses the organiser validator when utils/validate_submission.py exists and src/common/check_submission.py otherwise; the organiser script is never written or edited by us.
