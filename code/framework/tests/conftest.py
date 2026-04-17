"""Pytest bootstrap: make ``code/`` importable regardless of where pytest is run."""

import sys
from pathlib import Path

_REPO_CODE = Path(__file__).resolve().parents[2]
if str(_REPO_CODE) not in sys.path:
    sys.path.insert(0, str(_REPO_CODE))
