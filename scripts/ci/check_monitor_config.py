#!/usr/bin/env python3
"""Check monitor meta files for configuration issues.

This script scans monitor metadata JSON files for recorded configuration
issues and reports them with full details. Used in CI to detect problems
that should block the build.

Exit codes:
  0: No issues found
  1: Configuration issues detected
  2: Usage error or unexpected exception
"""

import json
import sys
from pathlib import Path
from typing import Optional


def check_monitor_config(pattern: Optional[str] = None) -> int:
    """Check monitor meta files for configuration issues.
    
    Parameters
    ----------
    pattern : str, optional
        Glob pattern for monitor meta files. If None, uses default locations.
    
    Returns
    -------
    int
        Exit code: 0 if OK, 1 if issues found, 2 if error.
    """
    if pattern is None:
        # Default search locations
        files = [Path("outputs/monitor_reports/monitor_meta.json")]
        # Also search in run directories
        import glob
        files.extend(
            Path(f) 
            for f in glob.glob("outputs/runs/*/monitor_reports/monitor_meta.json")
        )
    else:
        import glob
        files = [Path(f) for f in glob.glob(pattern)]
    
    if not files:
        print("⚠️  No monitor meta files found.")
        return 0
    
    issue_count = 0
    
    for filepath in files:
        try:
            if not filepath.exists():
                continue
                
            with filepath.open() as f:
                data = json.load(f)
            
            issues = data.get("config_issues", [])
            if issues:
                issue_count += len(issues)
                print(f"\n❌ Monitor config issues in {filepath}:")
                print(json.dumps(data, indent=2))
        
        except FileNotFoundError:
            # File was deleted between glob and read, skip
            continue
        except json.JSONDecodeError as e:
            print(f"⚠️  Could not parse {filepath}: {e}", file=sys.stderr)
            return 2
        except Exception as e:
            print(f"❌ Unexpected error reading {filepath}: {e}", file=sys.stderr)
            return 2
    
    if issue_count == 0:
        print("✅ No monitor config issues found")
        return 0
    else:
        print(f"\n❌ Found {issue_count} issue(s) in monitor meta files")
        return 1


def main() -> int:
    """Entry point for CLI."""
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Check monitor meta files for configuration issues."
    )
    parser.add_argument(
        "--pattern",
        type=str,
        default=None,
        help="Glob pattern for monitor meta files (default: outputs/monitor_reports/**)",
    )
    
    try:
        args = parser.parse_args()
        return check_monitor_config(pattern=args.pattern)
    except Exception as e:
        print(f"❌ Error: {e}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    sys.exit(main())
