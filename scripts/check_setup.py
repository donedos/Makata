"""Environment sanity check for Makata (thin wrapper — logic lives in app.cli).

Usage:
    python scripts/check_setup.py            # check only
    python scripts/check_setup.py --model    # also download/load XTTS v2
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.cli import main

if __name__ == "__main__":
    raise SystemExit(main(["check-setup", *sys.argv[1:]]))
