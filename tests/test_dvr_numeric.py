from __future__ import annotations

import unittest
from unittest.mock import patch

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

    def test_update_R0_keeps_integer_grid_ratios(self):
        dvr = self.DVR(
            n=self.np.array([14, 14, 14]),
            R0=self.np.array([3.0, 3.0, 7.2]),
            model="sho",
            symmetry=False,
            verbosity=0,
        )
        dvr.update_R0(
            self.np.array([6.04, 3.0, 7.2]),
            self.np.array([3.0, 3.0, 7.2]) / 14,
        )
        self.assertTrue(self.np.array_equal(dvr.n, self.np.array([28, 14, 14])))

    def test_sparse_solver_operator_has_fixed_shape_and_dtype(self):
        dvr = self.DVR(
            n=self.np.array([3, 0, 0]),
            R0=self.np.array([2.0, 0.0, 0.0]),
            model="sho",
            symmetry=False,
            sparse=True,
            verbosity=0,
        )

        def fake_eigsh(operator, k, which, v0):
            self.assertEqual(operator.shape, (7, 7))
            self.assertIsInstance(operator.shape[0], int)
            self.assertEqual(operator.dtype, self.np.dtype("float64"))
            result = operator.matvec(self.np.ones(operator.shape[1]))
            self.assertEqual(result.shape, (operator.shape[0],))
            return self.np.arange(k, dtype=float), self.np.zeros((operator.shape[0], k))

        with patch("HubbardTweezer.DVR.core.ssla.eigsh", side_effect=fake_eigsh):
            energies, states = dvr.H_solver(k=2)

        self.assertEqual(energies.shape, (2,))
        self.assertEqual(states.shape, (7, 2))

    def test_sparse_solver_ignores_wrong_length_initial_vector(self):
        dvr = self.DVR(
            n=self.np.array([3, 0, 0]),
            R0=self.np.array([2.0, 0.0, 0.0]),
            model="sho",
            symmetry=False,
            sparse=True,
            verbosity=0,
        )

        def fake_eigsh(operator, k, which, v0):
            self.assertIsNone(v0)
            return self.np.arange(k, dtype=float), self.np.zeros((operator.shape[0], k))

        with patch("HubbardTweezer.DVR.core.ssla.eigsh", side_effect=fake_eigsh):
            energies, states = dvr.H_solver(k=2, v0=self.np.ones(5))

        self.assertEqual(energies.shape, (2,))
        self.assertEqual(states.shape, (7, 2))

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
