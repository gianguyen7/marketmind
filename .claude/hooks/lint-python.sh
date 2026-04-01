#!/usr/bin/env bash
# Run ruff on staged Python files before commit
# Requires: pip install ruff

FILES=$(git diff --cached --name-only --diff-filter=ACM -- '*.py')
[ -z "$FILES" ] && exit 0

if command -v ruff &>/dev/null; then
  echo "$FILES" | xargs ruff check --select E,W,F --no-fix
  exit $?
else
  echo "ruff not installed, skipping lint (pip install ruff)"
  exit 0
fi
