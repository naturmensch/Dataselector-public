#!/usr/bin/env bash
# Conservative cleanup script for Dataselector
# - Dry-run by default
# - Produces a manifest (CSV + SHA256) for review
# - Moves candidates to a timestamped backup dir when --apply is given
# - Compresses backup and produces checksum
# - Designed to be conservative and safe

set -euo pipefail
IFS=$'\n\t'

progname=$(basename "$0")

usage() {
  cat <<EOF
Usage: $progname [--dry-run] [--apply] [--keep-days N] [--keep-count N] [--backup-dir DIR] [--compress] [--no-compress] [--interactive]

Conservative cleanup script (Dry‑Run default). Moves old runs, logs and experiment artifacts
from 'outputs' and root temporary files into a timestamped backup directory under 'data/archive/clean-backups'.

Options:
  --dry-run         (default) only show actions and write manifest; do not move/delete files
  --apply           actually perform move/compress actions (requires --confirm in interactive mode)
  --confirm         allow destructive operations when used with --apply
  --keep-days N     keep files newer than N days (default: 30)
  --keep-count N    keep N newest files per pattern (default: 3)
  --backup-dir DIR  base dir for backups (default: data/archive/clean-backups)
  --compress        compress backup dir into a tar.gz (default: enabled when --apply)
  --no-compress     do not compress even if --apply
  --interactive     ask for confirmation before applying destructive actions
  -h, --help        show this help

Examples:
  # Dry run (default)
  $progname --dry-run

  # Apply with default settings
  $progname --apply --confirm

EOF
}

# Defaults
DRY_RUN=1
APPLY=0
CONFIRM=0
KEEP_DAYS=30
KEEP_COUNT=3
BASE_BACKUP_DIR="data/archive/clean-backups"
COMPRESS=1
INTERACTIVE=0
INCLUDE_BACKUPS=0   # if 1, include outputs/backups in candidates
REMOVE_VENV=0       # if 1, remove venv and .venv when applying
LOGFILE="scripts/cleanup_workspace.log"

# Patterns to consider (conservative)
# Only search outside of data/archive and archive_local
PATTERNS=(
  "outputs/XXL_FULL_RUN_*.log"
  "outputs/*.pid"
  "outputs/pre_run_*.log"
  "outputs/pre_run_meta_*.json"
  "outputs/optuna_autoscale_best_*.json"
  "outputs/optuna_autoscale_report_*.md"
  "outputs/optuna_*_study_*.pkl"
  "outputs/*_summary_*.csv"
  "outputs/*.png"
  "outputs/*.pkl"
  "tmp_test.csv"
  "todoliste*.md"
  "*.pid"
)

# Helpers
echodo() {
  if [ "$DRY_RUN" -eq 1 ]; then
    echo "[DRY-RUN] $*"
  else
    echo "$*"
    eval "$*"
  fi
}

log() {
  echo "[$(date -u +'%Y-%m-%dT%H:%M:%SZ')] $*" | tee -a "$LOGFILE"
}

confirm_prompt() {
  local prompt="$1"
  if [ "$INTERACTIVE" -eq 1 ]; then
    read -r -p "$prompt [y/N]: " resp
    case "$resp" in
      [yY][eE][sS]|[yY]) return 0;;
      *) return 1;;
    esac
  fi
  # non-interactive returns true only if CONFIRM is set
  [ "$CONFIRM" -eq 1 ]
}

# Print helpful links/paths to thesis/manuscript artefacts if present
print_manuscript_info() {
  echo
  echo "--- Manuscript / Thesis links ---"
  # Prefer a summary produced by the thesis pipeline
  if [ -f "outputs/THESIS_XXL_SUMMARY.md" ]; then
    echo "Thesis summary: $(realpath --relative-to=. outputs/THESIS_XXL_SUMMARY.md)"
  fi
  if [ -f "outputs/THESIS_XXL_SUMMARY_$(date +%Y%m%d)_*.md" ]; then
    echo "Thesis summary (timestamped): see outputs/ for THESIS_XXL_SUMMARY_*.md"
  fi
  if [ -f "outputs/THESIS_FINAL_SELECTION_XXL.json" ]; then
    echo "Final selection (JSON): $(realpath --relative-to=. outputs/THESIS_FINAL_SELECTION_XXL.json)"
  fi
  if [ -f "docs/THESIS_MATERIALS.md" ]; then
    echo "Thesis materials / manuscript notes: $(realpath --relative-to=. docs/THESIS_MATERIALS.md)"
  fi
  # If repository has a remote, show a suggested GitHub path (best-effort)
  git_remote_url=""
  if git rev-parse --is-inside-work-tree >/dev/null 2>&1; then
    git_remote_url=$(git remote get-url origin 2>/dev/null || true)
  fi
  if [ -n "$git_remote_url" ]; then
    echo "Repository remote: $git_remote_url"
    echo "Suggested public link to thesis materials: <REPO_URL>/blob/main/docs/THESIS_MATERIALS.md (replace <REPO_URL> with the remote URL)"
  fi
  echo "--- End manuscript links ---"
  echo
}


parse_args() {
  while [ ${#:-} -gt 0 ]; do
    case "$1" in
      --dry-run) DRY_RUN=1; shift ;;
      --apply) APPLY=1; DRY_RUN=0; shift ;;
      --confirm) CONFIRM=1; shift ;;
      --keep-days) KEEP_DAYS="$2"; shift 2 ;;
      --keep-count) KEEP_COUNT="$2"; shift 2 ;;
      --backup-dir) BASE_BACKUP_DIR="$2"; shift 2 ;;
      --compress) COMPRESS=1; shift ;;
      --no-compress) COMPRESS=0; shift ;;
      --interactive) INTERACTIVE=1; shift ;;
      --include-backups) INCLUDE_BACKUPS=1; shift ;;
      --remove-venv) REMOVE_VENV=1; shift ;;
      -h|--help) usage; exit 0 ;;
      --) shift; break ;;
      -*) echo "Unknown option: $1"; usage; exit 2 ;;
      *) break ;;
    esac
  done
}

# Create backup dir
timestamp() { date +%Y%m%d_%H%M%S; }

main() {
  parse_args "$@"

  mkdir -p "$BASE_BACKUP_DIR"
  TS=$(timestamp)
  BACKUP_DIR="$BASE_BACKUP_DIR/$TS"
  mkdir -p "$BACKUP_DIR"

  log "Script started (dry-run=${DRY_RUN}, apply=${APPLY}, keep-days=${KEEP_DAYS}, keep-count=${KEEP_COUNT})"

  # Collect candidates (avoid data/archive and archive_local)
  candidates=()
  for pat in "${PATTERNS[@]}"; do
    while IFS= read -r -d '' f; do
      # ensure we don't touch files in data/archive, archive_local or raw data dirs
      case "$f" in
        data/archive/*|archive_local/*|data/images/*|data/raw/*) continue ;;
      esac
      candidates+=("$f")
    done < <(find . -path './data/archive' -prune -o -path './archive_local' -prune -o -path './data/images' -prune -o -path './data/raw' -prune -o -name "$(basename "$pat")" -print0 2>/dev/null)
  done

  # Add heuristic: include files in outputs/ that are older than KEEP_DAYS
  while IFS= read -r -d '' f; do
    candidates+=("$f")
  done < <(find outputs -type f -mtime +$KEEP_DAYS -print0 2>/dev/null)

  # Add pattern-based keep_count filtering: for XXL logs keep last KEEP_COUNT
  if compgen -G "outputs/XXL_FULL_RUN_*.log" > /dev/null; then
    mapfile -t all_xxl < <(ls -1t outputs/XXL_FULL_RUN_*.log 2>/dev/null || true)
    if [ "${#all_xxl[@]}" -gt "$KEEP_COUNT" ]; then
      for ((i=KEEP_COUNT;i<${#all_xxl[@]};i++)); do
        candidates+=("${all_xxl[$i]}")
      done
    fi
  fi

  # Deduplicate candidates
  mapfile -t uniq_candidates < <(printf '%s\n' "${candidates[@]}" | awk '!x[$0]++')

  # Filter out top-level dot folders, venvs and outputs/backups by default
  tmp_candidates=()
  for f in "${uniq_candidates[@]}"; do
    # normalize
    nf="${f#./}"
    # skip top-level dot folders (e.g., .git, .venv)
    if [[ "$nf" =~ ^\.[^/]+(/.*)?$ ]]; then
      continue
    fi
    # skip venv dirs
    if [[ "$nf" =~ ^(venv|\.venv)(/.*)?$ ]]; then
      continue
    fi
    # skip outputs/backups unless explicitly included
    if [[ "$nf" =~ ^outputs/backups(/.*)?$ ]] && [ "$INCLUDE_BACKUPS" -eq 0 ]; then
      continue
    fi
    tmp_candidates+=("$f")
  done
  uniq_candidates=("${tmp_candidates[@]}")

  # Write manifest to backup dir (even in dry-run)
  MANIFEST_CSV="$BACKUP_DIR/cleanup_manifest.csv"
  MANIFEST_SHA="$BACKUP_DIR/cleanup_manifest.sha256"
  : > "$MANIFEST_CSV"
  for f in "${uniq_candidates[@]}"; do
    if [ -e "$f" ]; then
      size=$(stat -c%s "$f" || echo 0)
      mtime=$(stat -c%y "$f" || echo "-")
      echo "$f,$size,$mtime" >> "$MANIFEST_CSV"
    fi
  done

  # Write sha256s for manifest entries
  cut -d, -f1 "$MANIFEST_CSV" | xargs -r -I{} sha256sum "{}" > "$MANIFEST_SHA" 2>/dev/null || true

  # Summary
  total_files=$(wc -l < "$MANIFEST_CSV" || echo 0)
  total_size=$(awk -F, '{s+=$2} END {print s+0}' "$MANIFEST_CSV" || echo 0)
  log "Candidate count: $total_files; total size: $total_size bytes"
  echo
  echo "Top 20 largest candidates:" 
  awk -F, '{print $2, $1}' "$MANIFEST_CSV" | sort -nr | head -n 20
  echo
  echo "Manifest written to: $MANIFEST_CSV"
  echo "SHA256s written to: $MANIFEST_SHA"

  # Provide clickable file link and quick-open hint
  if command -v realpath >/dev/null 2>&1; then
    ABS_MANIFEST=$(realpath "$MANIFEST_CSV")
  else
    ABS_MANIFEST="$PWD/$MANIFEST_CSV"
  fi
  echo "Manifest (file): file://$ABS_MANIFEST"
  echo "Open in VS Code: code -r '$MANIFEST_CSV'"
  echo "Or open with your editor: $MANIFEST_CSV"

  # Dry-run -> show what would be moved
  if [ "$DRY_RUN" -eq 1 ]; then
    echo
    echo "DRY‑RUN: no files are moved. To apply, re-run with --apply --confirm (or --apply --interactive)."
    echo
    # Show preview of moves
    if [ $total_files -gt 0 ]; then
      echo "Preview: files that would be moved to $BACKUP_DIR"
      awk -F, '{print $1}' "$MANIFEST_CSV"
    else
      echo "No candidate files found. Nothing to do."
    fi

    # Manuscript link info
    print_manuscript_info

    avail=$(df --output=avail -k "$BASE_BACKUP_DIR" | tail -1)
    avail_bytes=$((avail * 1024))
    if [ "$avail_bytes" -lt $(( total_size * 6 / 5 )) ]; then
      echo "Warning: not enough space in $BASE_BACKUP_DIR (avail: $avail_bytes bytes; needed ~ $(( total_size * 6 / 5 ))). Aborting."; exit 1
    fi

    # Optionally remove venv dirs permanently (not archived)
    if [ "$REMOVE_VENV" -eq 1 ]; then
      if confirm_prompt "Remove local venv directories (venv, .venv)? This deletes them permanently. Proceed?"; then
        echodo rm -rf venv .venv
        log "Removed venv directories"
      else
        echo "Skipping venv removal.";
      fi
    fi

    # Move files
    echo "Moving files to $BACKUP_DIR"
    while IFS= read -r -d '' file; do
      dest="$BACKUP_DIR/$(dirname "$file")"
      mkdir -p "$dest"
      echodo mv -- "$file" "$dest/"
    done < <(cut -d, -f1 "$MANIFEST_CSV" | sed 's/^\.\/\?//' | awk '{print $0"\0"}' ORS='')

    # Recreate manifest in backup dir root (paths relative inside backup)
    NEW_MANIFEST="$BACKUP_DIR/cleanup_manifest_postmove.csv"
    echo "path,size,mtime" > "$NEW_MANIFEST"
    (cd "$BACKUP_DIR" && find . -type f -printf '%P,%s,%TY-%Tm-%Td %TH:%TM:%TS\n') >> "$NEW_MANIFEST"

    if [ "$COMPRESS" -eq 1 ]; then
      tarball="$BACKUP_DIR.tar.gz"
      echo "Compressing backup to $tarball"
      tar -C "$BASE_BACKUP_DIR" -czf "$tarball" "$TS"
      sha256sum "$tarball" > "$tarball.sha256"
      echo "Compressed and checksummed: $tarball (sha256 in $tarball.sha256)"
    fi

    echo "Backup complete. Review $BACKUP_DIR and its checksums before deleting originals elsewhere."
    echo "To restore: tar -C / -xzf $tarball (inspect contents first)"
    log "Apply completed: moved $total_files files to $BACKUP_DIR"

    # Manuscript link info
    print_manuscript_info
  fi
}

main "$@"
