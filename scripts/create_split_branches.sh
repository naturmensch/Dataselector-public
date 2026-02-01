#!/usr/bin/env bash
# create_split_branches.sh
#
# Helps create split branches from a large branch by category.
# Uses file paths from docs/phase2_split_paths.md for reference.
#
# Usage: ./scripts/create_split_branches.sh <source_branch> [categories]
# Example: ./scripts/create_split_branches.sh refactor/e402-scripts-and-tests ci env docs tests scripts code
#
# Note: Manual verification required; doesn't auto-delete after merge.

set -euo pipefail

if [ $# -lt 1 ]; then
    echo "Usage: $0 <source_branch> [categories]"
    echo ""
    echo "Categories: ci env docs tests scripts code data other"
    echo ""
    echo "Example:"
    echo "  $0 refactor/e402-scripts-and-tests ci env tests"
    exit 1
fi

SOURCE_BRANCH="$1"
shift
CATEGORIES=("${@:-ci env docs tests scripts code}")

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

cd "$REPO_ROOT"

echo "🔄 Creating split branches from: $SOURCE_BRANCH"
echo "   Categories: ${CATEGORIES[*]}"
echo ""

# Ensure source branch exists
if ! git rev-parse --verify "$SOURCE_BRANCH" >/dev/null 2>&1; then
    echo "❌ Branch not found: $SOURCE_BRANCH"
    exit 1
fi

# Start from integration
git checkout integration -q
git pull --ff-only -q

CREATED=()

for category in "${CATEGORIES[@]}"; do
    split_branch="split/${SOURCE_BRANCH#*/}-${category}"
    
    echo "📦 Creating: $split_branch"
    
    # Create branch
    git checkout -b "$split_branch" "$SOURCE_BRANCH" -q 2>/dev/null || {
        echo "   (Branch exists, checking out)"
        git checkout "$split_branch" -q
    }
    
    # File patterns per category (from phase2_split_paths.md conceptually)
    case "$category" in
        ci)
            patterns=(".github/")
            ;;
        env)
            patterns=("Makefile" "environment.yml" "pyproject.toml" "pytest.ini" "requirements.txt" "mypy.ini" "setup.cfg")
            ;;
        docs)
            patterns=("docs/" "README.md" "CHANGELOG.md" "CONTRIBUTING.md" "README_*.md")
            ;;
        tests)
            patterns=("tests/")
            ;;
        scripts)
            patterns=("scripts/")
            ;;
        code)
            patterns=("src/" "dataselector/")
            ;;
        data)
            patterns=("data/" "outputs/")
            ;;
        other)
            patterns=("config/" ".flake8" ".pre-commit-config.yaml" "contrib/" "locks/" "tools/")
            ;;
        *)
            echo "   ⚠️  Unknown category: $category (skipping)"
            continue
            ;;
    esac
    
    echo "   Files: ${patterns[*]}"
    echo ""
    
    # Note: Actual file filtering would happen during merge
    # (this script just creates the branches; filtering via git restore in merge workflow)
    
    CREATED+=("$split_branch")
done

echo ""
echo "✅ Created split branches:"
printf '   - %s\n' "${CREATED[@]}"
echo ""
echo "📝 Next steps:"
echo "   1. For each split branch, verify it contains only desired files:"
echo "      git diff --name-only integration..split/branch"
echo "   2. Merge one by one with gates: ./scripts/validate_merge_gate.sh ..."
echo "   3. After successful merge of all splits, delete branch:"
echo "      git branch -d split/branch"
echo ""
