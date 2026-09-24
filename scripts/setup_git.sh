#!/usr/bin/env bash
# Run once per laptop from the repo root:  bash scripts/setup_git.sh "<Name>" "<github email>"
set -e
[ -n "$1" ] && [ -n "$2" ] || { echo "usage: bash scripts/setup_git.sh \"<Name>\" \"<github email>\""; exit 1; }
git config user.name "$1"
git config user.email "$2"
git config core.hooksPath .githooks
# tool-specific files never enter the tree; excluded locally, never via the committed .gitignore
cat >> .git/info/exclude <<'X'
CLAUDE.md
.claude/
.cursor/
.copilot*
AGENTS.md
.aider*
.continue/
X
echo "identity: $(git config user.name) <$(git config user.email)>; hooks: $(git config core.hooksPath)"
