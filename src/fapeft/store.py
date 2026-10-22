"""Crash-safe persistence: every result is written atomically and cached by path.

A re-run of any stage skips work whose output file already exists, so the whole
pipeline can be killed at any moment and restarted without losing finished work.
"""
import json
import os
from pathlib import Path
from typing import Any, Callable


def atomic_write_text(path: Path, text: str) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.tmp")
    tmp.write_text(text)
    os.replace(tmp, path)  # atomic on POSIX: readers see the old or the new file, never half of one


def write_json(path: Path, obj: Any) -> None:
    atomic_write_text(path, json.dumps(obj, indent=2))


def read_json(path: Path) -> Any:
    """Returns None when the file is missing or unreadable (treated as 'not computed')."""
    try:
        return json.loads(Path(path).read_text())
    except (FileNotFoundError, json.JSONDecodeError):
        return None


def cached_json(path: Path, fn: Callable[[], Any], log: Callable[[str], None] = print) -> Any:
    hit = read_json(path)
    if hit is not None:
        log(f"  cache hit: {path.name}")
        return hit
    value = fn()
    write_json(path, value)
    log(f"  saved:     {path.name}")
    return value
