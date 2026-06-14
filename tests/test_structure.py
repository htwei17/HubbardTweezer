from __future__ import annotations

import ast
from pathlib import Path
import unittest

from tests._helpers import PACKAGE_ROOT, ROOT


class TestSourceLayout(unittest.TestCase):
    def test_const_module_removed(self):
        self.assertFalse((PACKAGE_ROOT / "DVR" / "const.py").exists())

    def test_release_excludes_dynamics_plot_io_and_dvr_cli_modules(self):
        blocked = [
            PACKAGE_ROOT / "DVR" / "cli.py",
            PACKAGE_ROOT / "DVR" / "dynamics.py",
            PACKAGE_ROOT / "DVR" / "io.py",
            PACKAGE_ROOT / "DVR" / "plot.py",
            PACKAGE_ROOT / "Hubbard" / "dynamics.py",
            PACKAGE_ROOT / "Hubbard" / "plot.py",
            PACKAGE_ROOT / "tools" / "display_top.py",
            PACKAGE_ROOT / "tools" / "dynamics.py",
            PACKAGE_ROOT / "tools" / "graph.py",
        ]
        self.assertEqual([str(path) for path in blocked if path.exists()], [])

    def test_no_source_imports_legacy_const_module(self):
        offenders: list[str] = []
        for path in PACKAGE_ROOT.rglob("*.py"):
            tree = ast.parse(path.read_text(encoding="utf-8"))
            for node in ast.walk(tree):
                if isinstance(node, ast.ImportFrom) and node.module:
                    if node.module == "const" or node.module.endswith(".const"):
                        offenders.append(str(path.relative_to(ROOT)))
                elif isinstance(node, ast.Import):
                    for alias in node.names:
                        if alias.name == "const" or alias.name.endswith(".const"):
                            offenders.append(str(path.relative_to(ROOT)))
        self.assertEqual(offenders, [])

    def test_metadata_defines_shared_constants(self):
        metadata_path = PACKAGE_ROOT / "DVR" / "metadata.py"
        tree = ast.parse(metadata_path.read_text(encoding="utf-8"))
        names = set()
        for node in tree.body:
            if isinstance(node, ast.Assign):
                for target in node.targets:
                    if isinstance(target, ast.Name):
                        names.add(target.id)
        self.assertTrue({"A0", "AMU", "DIM", "h"}.issubset(names))

    def test_public_api_exports_shared_constants(self):
        dvr_init = (PACKAGE_ROOT / "DVR" / "__init__.py").read_text(encoding="utf-8")
        root_init = (PACKAGE_ROOT / "__init__.py").read_text(encoding="utf-8")
        for name in ("A0", "AMU", "DIM", "h"):
            self.assertIn(name, dvr_init)
            self.assertIn(name, root_init)

    def test_hubbard_exe_is_thin_wrapper(self):
        script = (ROOT / "src" / "Hubbard_exe.py").read_text(encoding="utf-8")
        self.assertIn("from HubbardTweezer.Hubbard.cli import main", script)
        self.assertIn("raise SystemExit(main())", script)

    def test_hubbard_cli_does_not_import_plot_at_module_load(self):
        cli_path = PACKAGE_ROOT / "Hubbard" / "cli.py"
        tree = ast.parse(cli_path.read_text(encoding="utf-8"))
        top_level_plot_imports = [
            node
            for node in tree.body
            if isinstance(node, ast.ImportFrom)
            and node.module == "plot"
            and node.level > 0
        ]
        self.assertEqual(top_level_plot_imports, [])

    def test_public_api_excludes_unreleased_symbols(self):
        dvr_init = (PACKAGE_ROOT / "DVR" / "__init__.py").read_text(encoding="utf-8")
        hubbard_init = (PACKAGE_ROOT / "Hubbard" / "__init__.py").read_text(
            encoding="utf-8"
        )
        root_init = (PACKAGE_ROOT / "__init__.py").read_text(encoding="utf-8")
        for source in (dvr_init, hubbard_init, root_init):
            for name in (
                "DVRDynamics",
                "DVRDynamicsIO",
                "DVRPlot",
                "LatticeDynamics",
                "HubbardGraph",
            ):
                self.assertNotIn(name, source)
