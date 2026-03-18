from __future__ import annotations

from pathlib import Path


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def exists(path: Path) -> bool:
    return path.exists() and path.is_file()
