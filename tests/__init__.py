"""Ensure project root is on sys.path when running tests directly.

Running a test file as a script (e.g. ``python tests/test_foo.py``) adds the
``tests`` directory to ``sys.path`` but **not** the project root.  Therefore
imports like ``from src.config import Config`` fail with
``ModuleNotFoundError: No module named 'src'``.

We fix this once for all tests by inserting the parent directory (project root)
into ``sys.path`` if it is missing.
"""
from __future__ import annotations

import pathlib
import sys

PROJECT_ROOT = pathlib.Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    # Prepend to give project modules precedence over site-packages in case of
    # name collisions.
    sys.path.insert(0, str(PROJECT_ROOT))
