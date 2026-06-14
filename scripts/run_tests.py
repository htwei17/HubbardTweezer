#!/usr/bin/env python3
from __future__ import annotations

import compileall
from importlib.util import find_spec
from pathlib import Path
import sys
import unittest


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
PACKAGE = SRC / "HubbardTweezer"
OPTIONAL_DEPS = (
    "configobj",
    "h5py",
    "nlopt",
    "numpy",
    "opt_einsum",
    "scipy",
)


def compile_sources() -> bool:
    ok = compileall.compile_dir(str(PACKAGE), quiet=1)
    ok = compileall.compile_file(str(SRC / "Hubbard_exe.py"), quiet=1) and ok
    return ok


def report_optional_dependencies() -> None:
    missing = [name for name in OPTIONAL_DEPS if find_spec(name) is None]
    if not missing:
        print("Optional dependency check: all test dependencies are available.")
        return
    print("Optional dependency check: missing " + ", ".join(missing))
    print("Numerical tests that require those packages will be skipped.")


def run_unittests() -> unittest.result.TestResult:
    loader = unittest.defaultTestLoader
    suite = loader.discover(start_dir=str(ROOT / "tests"), top_level_dir=str(ROOT))
    runner = unittest.TextTestRunner(verbosity=2)
    return runner.run(suite)


def main() -> int:
    sys.path.insert(0, str(ROOT))
    report_optional_dependencies()
    compiled = compile_sources()
    if not compiled:
        print("compileall failed.")
    result = run_unittests()
    return 0 if compiled and result.wasSuccessful() else 1


if __name__ == "__main__":
    raise SystemExit(main())
