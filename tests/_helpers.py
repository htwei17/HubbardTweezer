from __future__ import annotations

from contextlib import contextmanager
from importlib.util import find_spec
from pathlib import Path
import os
import sys


ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = ROOT / "src"
PACKAGE_ROOT = SRC_ROOT / "HubbardTweezer"


def add_src_to_path() -> None:
    src = str(SRC_ROOT)
    if src not in sys.path:
        sys.path.insert(0, src)


def missing_modules(*modules: str) -> list[str]:
    return [module for module in modules if find_spec(module) is None]


def has_modules(*modules: str) -> bool:
    return not missing_modules(*modules)


def missing_message(*modules: str) -> str:
    missing = missing_modules(*modules)
    if not missing:
        return ""
    return "missing optional dependencies: " + ", ".join(missing)


@contextmanager
def working_directory(path) -> None:
    previous = Path.cwd()
    os.chdir(path)
    try:
        yield
    finally:
        os.chdir(previous)
