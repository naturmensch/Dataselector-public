#!/usr/bin/env python3
"""Compatibility wrapper for `dataselector validate-kmeans`."""

from __future__ import annotations

import subprocess
import sys


def main() -> int:
    cmd = [sys.executable, "-m", "dataselector", "validate-kmeans", *sys.argv[1:]]
    return subprocess.call(cmd)


if __name__ == "__main__":
    raise SystemExit(main())
