#!/usr/bin/env bash
# validate_merge_gate.sh
#
# Validates that merge gates are satisfied after each batch/merge.
# Prevents proceeding with broken state.
#
# Usage: ./scripts/validate_merge_gate.sh [batch_name] [extra_tests]
# Example: ./scripts/validate_merge_gate.sh "Batch A" "make test"
#          ./scripts/validate_merge_gate.sh "Batch B" "make check-env && make test"

set -euo pipefail

BATCH_NAME="${1:-General Gate}"
EXTRA_TESTS="${2:-make test}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

cd "$REPO_ROOT"

echo ""
echo "╔════════════════════════════════════════════════════════════════╗"
echo "║             MERGE GATE VALIDATION: $BATCH_NAME                 ║"
echo "╚════════════════════════════════════════════════════════════════╝"
echo ""

PASS=true

# 1. Git status check
echo "📋 [1/5] Checking git status..."
if ! git status --short | grep -q "^UU\|^AA\|^DD"; then
    echo "    ✅ No merge conflicts"
else
    echo "    ❌ Merge conflicts detected! Resolve before proceeding."
    git status --short | grep "^UU\|^AA\|^DD"
    PASS=false
fi

# 2. Verify last merge commit
echo "📋 [2/5] Verifying merge commit..."
last_msg=$(git log -1 --oneline)
if [[ "$last_msg" =~ "Merge branch" ]]; then
    echo "    ✅ Last commit is merge: $last_msg"
else
    echo "    ⚠️  Last commit not a merge: $last_msg"
fi

# 3. Run tests
echo "📋 [3/5] Running tests: $EXTRA_TESTS"
if eval "$EXTRA_TESTS"; then
    echo "    ✅ Tests passed"
else
    echo "    ❌ Tests FAILED! Fix before continuing."
    PASS=false
fi

# 4. Check for large untracked files
echo "📋 [4/5] Checking for accidental large artifacts..."
untracked_size=$(find . -type f -path "./.git" -prune -o \
    \( -type f -size +10M -print \) 2>/dev/null | wc -l)
if [ "$untracked_size" -eq 0 ]; then
    echo "    ✅ No large untracked files"
else
    echo "    ⚠️  Found $untracked_size large files:"
    find . -type f -path "./.git" -prune -o \
        \( -type f -size +10M -print \) 2>/dev/null | head -5
    echo "    (Check if these should be in .gitignore)"
fi

# 5. Summary
echo "📋 [5/5] Gate summary..."
echo ""

if [ "$PASS" = true ]; then
    echo "╔════════════════════════════════════════════════════════════════╗"
    echo "║  ✅ GATE PASSED: Safe to proceed with next merge/batch        ║"
    echo "╚════════════════════════════════════════════════════════════════╝"
    echo ""
    return 0 2>/dev/null || exit 0
else
    echo "╔════════════════════════════════════════════════════════════════╗"
    echo "║  ❌ GATE FAILED: Fix issues before proceeding                 ║"
    echo "╚════════════════════════════════════════════════════════════════╝"
    echo ""
    return 1 2>/dev/null || exit 1
fi
