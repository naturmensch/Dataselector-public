#!/usr/bin/env bash
# check_branch_supersets.sh
#
# Identifies stacked branches (where branch B contains all commits of branch A)
# Useful for detecting dependencies before Phase 2 merging.
#
# Usage: ./scripts/check_branch_supersets.sh [base_ref]
# Example: ./scripts/check_branch_supersets.sh origin/main
#
# Output: For each pair (A, B), prints if B is a superset of A

set -euo pipefail

BASE_REF="${1:-origin/main}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

cd "$REPO_ROOT"

# Fetch latest
echo "🔄 Fetching latest branch info..."
git fetch --all --prune -q 2>/dev/null || true

# Get all remote branches (excluding origin/HEAD, origin/main, origin/integration)
echo "📋 Analyzing branch relationships..."
echo ""

BRANCHES=($(git branch -r --format='%(refname:short)' | grep -v "HEAD\|/main\|/integration" | sort))

if [ ${#BRANCHES[@]} -eq 0 ]; then
    echo "❌ No branches found (excluding main, integration, HEAD)"
    exit 1
fi

echo "Found ${#BRANCHES[@]} branches to analyze."
echo ""
echo "=== SUPERSET ANALYSIS ==="
echo ""

STACKED=()
INDEPENDENT=()

for i in "${!BRANCHES[@]}"; do
    for j in "${!BRANCHES[@]}"; do
        if [ "$i" -ne "$j" ]; then
            b1="${BRANCHES[$i]}"
            b2="${BRANCHES[$j]}"
            
            # Check if b1 is ancestor of b2 (i.e., b2 contains all of b1)
            if git merge-base --is-ancestor "$b1" "$b2" 2>/dev/null; then
                echo "⚠️  STACKED: $b2 contains all commits from $b1"
                STACKED+=("$b1 ← $b2")
            fi
        fi
    done
done

echo ""
echo "=== SUMMARY ==="
if [ ${#STACKED[@]} -gt 0 ]; then
    echo ""
    echo "🔗 Stacked branches found (${#STACKED[@]}):"
    printf '%s\n' "${STACKED[@]}"
    echo ""
    echo "⚡ Recommendation: For each pair (A ← B),"
    echo "   - Decide: merge only B (drop A), or split B before merging both"
    echo "   - Document in conflict log before Phase 2"
else
    echo "✅ No stacked branches detected (all branches are independent)"
fi

echo ""
echo "=== DETAILED BRANCH INFO ==="
echo ""

for branch in "${BRANCHES[@]}"; do
    commits=$(git rev-list --count "$BASE_REF..$branch" 2>/dev/null || echo "?")
    files=$(git diff --name-only "$BASE_REF..$branch" 2>/dev/null | wc -l)
    last_author=$(git log -1 --format='%an' "$branch" 2>/dev/null || echo "?")
    last_date=$(git log -1 --format='%ai' "$branch" 2>/dev/null | cut -d' ' -f1 || echo "?")
    
    printf "%-40s | commits: %3s | files: %3d | last: %s by %s\n" \
        "$branch" "$commits" "$files" "$last_date" "$last_author"
done

echo ""
echo "💡 Tip: Use this output to plan your merge batches and identify splits."
