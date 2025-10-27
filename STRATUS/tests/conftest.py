"""Test configuration for STRATUS unit tests."""

from __future__ import annotations

import sys
from pathlib import Path

# Ensure the project root is importable when running the ``pytest`` console script.
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
