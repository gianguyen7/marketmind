#!/usr/bin/env bash
# Scan staged Python files for common temporal leakage patterns
# Used as a pre-commit hook — warns but does not block

LEAKAGE_PATTERNS=(
  'train_test_split'
  'shuffle=True'
  'KFold('
  'StratifiedKFold('
  "\.transform\('mean'\)"
)

FILES=$(git diff --cached --name-only --diff-filter=ACM -- '*.py')
[ -z "$FILES" ] && exit 0

FOUND=0
for pattern in "${LEAKAGE_PATTERNS[@]}"; do
  MATCHES=$(echo "$FILES" | xargs grep -ln "$pattern" 2>/dev/null)
  if [ -n "$MATCHES" ]; then
    echo "⚠ Potential temporal leakage pattern '$pattern' in:"
    echo "$MATCHES" | sed 's/^/  /'
    FOUND=1
  fi
done

if [ "$FOUND" -eq 1 ]; then
  echo ""
  echo "Review these patterns. Use TimeSeriesSplit and time-based splits only."
  echo "If intentional (e.g., in test utilities), proceed with commit."
fi

exit 0  # warn only, don't block
