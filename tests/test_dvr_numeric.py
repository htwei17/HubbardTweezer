from __future__ import annotations

import unittest

from tests._helpers import add_src_to_path, has_modules, missing_message


DVR_DEPS = ("numpy", "opt_einsum", "scipy")


@unittest.skipUnless(has_modules(*DVR_DEPS), missing_message(*DVR_DEPS))
class TestDVRNumerics(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        add_src_to_path()
        import numpy as np

        from HubbardTweezer.DVR import DVR, DVRConfig

        cls.np = np
        cls.DVR = DVR
        cls.DVRConfig = DVRConfig

    def test_metadata_roundtrip(self):
        cfg = self.DVRConfig.from_inputs(
            n=[6, 0, 0],
            R0=[4.0, 0.0, 0.0],
            model="sho",
            trap=(1.0, 1.0),
            symmetry=False,
            verbosity=0,
        )
        dvr = self.DVR.from_metadata(cfg)
        self.assertEqual(dvr.metadata.config.n, cfg.n)
        self.assertEqual(dvr.metadata.grid.n, cfg.n)
        self.assertEqual(dvr.metadata.config.trap.waist_nm, cfg.trap.waist_nm)

    def test_free_potential_returns_zero(self):
        dvr = self.DVR(
            n=self.np.array([4, 0, 0]),
            R0=self.np.array([2.0, 0.0, 0.0]),
            avg=0,
            model="sho",
            symmetry=False,
            verbosity=0,
        )
        value = float(dvr.Vfun(0.0, 0.0, 0.0))
        self.assertEqual(value, 0.0)

    def test_sho_spectrum_matches_low_lying_levels(self):
        dvr = self.DVR(
            n=self.np.array([12, 0, 0]),
            R0=self.np.array([6.0, 0.0, 0.0]),
            model="sho",
            symmetry=False,
            verbosity=0,
        )
        energies, _ = dvr.H_solver(k=3)
        expected = self.np.array([0.5, 1.5, 2.5])
        self.assertTrue(self.np.all(self.np.diff(energies) > 0))
        self.assertLess(self.np.max(self.np.abs(energies - expected)), 0.35)
