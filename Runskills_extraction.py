"""
Backward-compatible launcher for skills extraction v2.

The application lives under `skills_extraction/`. Prefer:
  python -m skills_extraction --input ... --output-dir ...

This file only adds the repo root to sys.path and delegates to the package CLI.
"""

from __future__ import annotations

import sys
from pathlib import Path

# Windows UTF-8 console (match package CLI behavior)
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

_ROOT = Path(__file__).resolve().parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from skills_extraction.cli import main  # noqa: E402

if __name__ == "__main__":
    main()
