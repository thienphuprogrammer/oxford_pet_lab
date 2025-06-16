from __future__ import annotations

"""file_utils.py
Utility helpers for common filesystem I/O used across the project.
All helpers expose a simple functional interface and take care of creating
parent directories whenever necessary.
"""

from typing import Any
import pathlib
import json
import pickle

__all__ = [
    "timestamp",
    "save_json",
    "load_json",
    "save_pickle",
    "load_pickle",
]


def timestamp() -> str:
    """Return an ISO-like timestamp suitable for filenames (YYYYmmdd_HHMMSS)."""
    from datetime import datetime

    return datetime.now().strftime("%Y%m%d_%H%M%S")


# -----------------------------------------------------------------------------
# JSON helpers
# -----------------------------------------------------------------------------


def _prepare_path(path: str | pathlib.Path) -> pathlib.Path:
    """Convert argument to :class:`pathlib.Path` and create parent folder."""
    p = pathlib.Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    return p


def save_json(obj: Any, path: str | pathlib.Path) -> None:
    """Serialize *obj* as pretty-printed JSON to *path*.

    Parameters
    ----------
    obj
        JSON-serialisable python object.
    path
        Destination filepath. Parent folders will be created automatically.
    """
    p = _prepare_path(path)
    with p.open("w", encoding="utf-8") as f:
        # Ensure objects like pathlib.Path are serialized as strings
        json.dump(obj, f, indent=2, ensure_ascii=False, default=str)


def load_json(path: str | pathlib.Path) -> Any:
    """Load JSON file at *path* and return python object.

    If the file does not exist this returns an empty ``dict``.
    """
    p = pathlib.Path(path)
    if not p.exists():
        return {}
    with p.open("r", encoding="utf-8") as f:
        return json.load(f)


# -----------------------------------------------------------------------------
# Pickle helpers (used for caching intermediate results)
# -----------------------------------------------------------------------------


def save_pickle(obj: Any, path: str | pathlib.Path) -> None:
    """Pickle *obj* to *path* with gzip compression if ``.gz`` extension is used."""
    import gzip

    p = _prepare_path(path)
    open_fn = gzip.open if p.suffix == ".gz" else open
    with open_fn(p, "wb") as f:
        pickle.dump(obj, f)


def load_pickle(path: str | pathlib.Path) -> Any:
    """Load pickle file and return python object. Handles optional ``.gz``."""
    import gzip

    p = pathlib.Path(path)
    if not p.exists():
        raise FileNotFoundError(f"File not found: {path}")
    open_fn = gzip.open if p.suffix == ".gz" else open
    with open_fn(p, "rb") as f:
        return pickle.load(f)
