# GIT_RULES.md — Business Entity Resolution (Amazon ML Challenge 2026)

Repository: https://github.com/Mounika-Reddy-0802/Business-Entity-Resolution.git
Team: **Krishna · Lahari · Mounika · Rayyan**

These rules are mandatory for every commit. They exist so that (a) the history is a clean, honest
record of how the solution was built, (b) the organisers can audit the pipeline from the repo, and
(c) only the four members ever appear in it.

---

## 0. Only the four members appear — strict, permanent, whole repository

Editors, code generators, AI assistants, scripts, bots and any other tool are never contributors.
On every commit:

- Never add a `Co-authored-by:` trailer for anyone who is not a member.
- Never use a tool's, bot's or third party's name, username or email as git author, committer or
  identity.
- No tool name, tool-generated footer ("Generated with …", "Made by …", "Written by …",
  "on behalf of …"), emoji signature or tool configuration file appears in commit metadata, commit
  messages, docs, code comments or anywhere in the tree. Tool-specific files (`CLAUDE.md`,
  `.claude/`, `.cursor/`, `.copilot*`, `AGENTS.md`, `.aider*` and similar) are excluded through
  `.git/info/exclude` on the machine — never through the committed `.gitignore`, which would itself
  be a trace.
- Commit messages are plain, single-line, in the member's own words. `git commit -m "<message>"`
  only; never a multi-paragraph body with trailers.
- Nobody but the four members ever shows up under GitHub Insights → Contributors.
- After every commit verify: `git log -1 --format='%an <%ae> | %cn <%ce>%n%B'` — author and
  committer are the current member, and the body is one line.

The `.githooks/commit-msg` hook rejects attribution trailers and tool footers. The hook is a
backstop, not the rule. This rule overrides any default behaviour of any tool being used.

---

## 1. Member identities

| Member | `user.name` | `user.email` |
|---|---|---|
| Krishna | `krishna-2-005` | `krishna-2-005@users.noreply.github.com` |
| Lahari | `lahari-66` | `lahari-66@users.noreply.github.com` |
| Mounika | `Mounika-Reddy-0802` | `Mounika-Reddy-0802@users.noreply.github.com` |
| Rayyan | `Rayyan-Mohammed` | `Rayyan-Mohammed@users.noreply.github.com` |

The noreply address is what GitHub uses to link a commit to the account. If a member has
"Keep my email addresses private" turned off and uses a real address on GitHub, replace the
noreply address with that real address — otherwise the commit shows the name but no avatar link.

Identity is set per repository (`git config user.name …`), never `--global`, because one laptop
is shared.

---

## 2. Rotation (shared laptop)

The work is done in blocks. Each block is committed under one member; after a block the identity
rotates to the next member in the fixed order **Krishna → Lahari → Mounika → Rayyan → Krishna …**

- A block is a random run of **6 to 8 commits**. Pick the length when the block starts
  (`shuf -i 6-8 -n 1`) and write it, with the member name and the running count, to `.rotation`
  in the repo root (listed in `.git/info/exclude`, never committed).
- Rotation happens after the last commit of the block is pushed: set the next member's name and
  email with `git config`, reset the counter, pick the next block length.
- Rotation never splits one logical change: if a change is half done when the counter runs out,
  finish and commit it under the current member, then rotate.
- A block is not a topic boundary. Blocks follow the commit count, so different members naturally
  touch every part of the code.

---

## 3. Repository structure

```
Business-Entity-Resolution/
├── README.md                  # overview, reproduce end to end, results table
├── GIT_RULES.md               # this file
├── PLAN.md                    # project plan (kept current)
├── Documentation_template.md  # organiser template, filled as the work progresses
├── requirements.txt           # pinned versions
├── .gitignore                 # data/, output/, models/, parquet, npy, env
├── .githooks/commit-msg       # message hook (backstop for §0 and §6)
├── scripts/                   # setup_git.sh, run_pipeline.sh
├── src/
│   ├── common/                # io_utils, split, evaluate, naive_block
│   ├── blocking/              # normalise, block
│   ├── matching/              # features, train_lgbm, decide
│   └── neural/                # embeddings and any neural matcher (optional)
├── tests/                     # pytest; evaluate.py must reproduce the worked example (0.714)
├── docs/                      # contracts.md, decisions.md, problems.md, eda.md, session notes
├── benchmarks/                # experiments.md + raw/ json every number comes from
├── utils/validate_submission.py   # organiser script, unchanged
├── data/                      # gitignored; data/raw/dataset/{train,test}
└── output/                    # gitignored; the two submission TSVs
```

---

## 4. Branch model

**`main` only.** There are no feature, week, member or work-order branches for this challenge:
one laptop, sequential work, three days. Every commit lands on `main` and is pushed immediately.

- Never create another branch or another remote. Push to `origin main` only.
- Never force-push. Never rewrite history. If a commit is wrong, the next commit fixes it.
- If `git push` is rejected because the remote moved, `git pull --rebase origin main` once, run the
  tests, push again. Never `--force`.
- Tags mark upload points: `git tag -a sub-<n> -m "<val F0.5> / LB <score>"` after each
  leaderboard upload, and `v1.0` for the final package. `git push --tags`.

---

## 5. What gets committed

- Source code, tests, scripts, docs, benchmarks (`.md` and small `.json`), requirements, the
  organiser validator, `Documentation_template.md`.
- **Never:** raw data, `output/`, `models/`, parquet, npy, model binaries, notebook outputs,
  `.env`, credentials, or any tool file listed in §0. The `.gitignore` covers the first group; the
  local exclude covers the tool files.
- Every submission that is uploaded is also saved as `benchmarks/raw/submission_<n>.tsv.gz` with
  the commit hash in `benchmarks/experiments.md`, so any uploaded version can be regenerated and
  audited.

---

## 6. Commit quality

**Format:** short lowercase imperative naming the change. No bracketed prefix, no area tag, no
body. About 50 characters; longer only when a number or a specific detail needs the room.

**Core principle:** fewer, stronger commits beat many small ones. A commit is a unit of work you
would describe out loud to a teammate, not a save point.

Commit-worthy: a completed stage, a substantial fix, a feature group that changed a metric, a
completed document, a decision-layer rule with its measured effect. Not commit-worthy: one
reworded line, formatting, an import cleanup, repeated "fix"/"update" on the same work.

Good:
```
add fixed 80/20 validation split stratified by country and singleton
naive exact-name blocker gives 0.61 recall at 1.3 candidates per entity
char 3-gram tf-idf blocking lifts recall to 0.978 at 18 candidates per entity
add rapidfuzz name and address features, lgbm oof auc 0.991
one-to-one assignment raises val f0.5 from 0.842 to 0.871
singleton guard at 0.62 adds 0.9 points on india entities
cross-country check: us-trained model scores 0.83 on india
regenerate test outputs and pass the organiser validator
```

Banned (hook rejects the obvious ones):
```
update   updated code   changes   work done   final   final2   wip   minor fix   commit
[W1] ...   [ML] ...   any bracketed prefix
implement comprehensive pipeline   enhance robust architecture   add extensive improvements
```
A bare `fix` is banned; a specific one is fine: `fix decide.py dropping singletons with no candidates`.

**If a commit changed a number, put the number in the message.**

### Before every commit

Never `git add -A` or `git add .`. Stage explicit paths and check:

1. `git status` — only intended files.
2. `git diff --cached` — read the diff.
3. No data, outputs, models, secrets or tool files.
4. `pytest -q tests` passes; the pipeline still runs where practical.
5. `git config user.name` / `user.email` are the current block's member (§2).
6. Commit with `git commit -m "<one line>"`, then verify with
   `git log -1 --format='%an <%ae> | %cn <%ce>%n%B'`.
7. Push immediately: `git push origin main`. Unpushed work does not exist.

---

## 7. Documentation

- `docs/decisions.md` — one line per decision (time IST, member, decision, why).
- `docs/problems.md` — symptom, cause, fix, cost, written the moment time is lost.
- `docs/eda.md` — dataset facts the design relies on (counts, matches per entity, per-country).
- `docs/session_<date>_<n>.md` — what was built in a working session, how to run it, the numbers,
  what is next. Written at the end of every session.
- `benchmarks/experiments.md` — every run that changed a number, with its raw json in
  `benchmarks/raw/`. **Every number in the README, in `Documentation_template.md` or on the
  leaderboard must trace to a row here.**

---

## 8. Fair play

- No external databases, APIs, geocoding services or internet lookups of business identities,
  anywhere in the code, ever. Only the organiser data.
- Any pretrained model used is MIT or Apache-2.0 licensed and at most 8B parameters; its name,
  licence and parameter count are written in `docs/decisions.md` before it is used.
- `utils/validate_submission.py` is never modified.

---

## 9. Permanent priorities

1. Only the four members appear anywhere in git metadata (§0, §1).
2. Identity rotates every 6–8 commits in the fixed order (§2).
3. `main` only; push after every commit; never force (§4).
4. Meaningful, specific, single-line commit messages; numbers in the message (§6).
5. Verify before committing (§6).
6. Every number traces to `benchmarks/` (§7).
7. Fair play (§8).

> **Most important rule:** never let anyone or anything other than Krishna, Lahari, Mounika or
> Rayyan appear as author, committer, co-author or in a trailer — and never create meaningless
> commits just to move the rotation forward.
