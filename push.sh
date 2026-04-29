#!/usr/bin/env bash
# Stage all changes, commit with a message you provide (or a default), then push.
# Usage:
#   ./push.sh                       # commits with timestamp message, pushes current branch
#   ./push.sh "your commit message" # uses your message
#   ./push.sh "msg" main            # commit + push to "main"

set -euo pipefail

cd "$(dirname "$0")"

MSG="${1:-Update $(date '+%Y-%m-%d %H:%M:%S')}"
BRANCH="${2:-$(git rev-parse --abbrev-ref HEAD)}"

if ! git rev-parse --is-inside-work-tree >/dev/null 2>&1; then
  echo "Not a git repo." >&2
  exit 1
fi

if ! git remote get-url origin >/dev/null 2>&1; then
  echo "No 'origin' remote configured. Run: git remote add origin <url>" >&2
  exit 1
fi

git add -A

if git diff --cached --quiet; then
  echo "No staged changes to commit. Pushing branch as-is..."
else
  git commit -m "$MSG"
fi

git push -u origin "$BRANCH"
echo "Pushed $BRANCH to origin."
