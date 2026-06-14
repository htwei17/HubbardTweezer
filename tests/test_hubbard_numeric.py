from __future__ import annotations

import unittest

from tests._helpers import add_src_to_path, has_modules, missing_message


HUBBARD_DEPS = ("numpy", "opt_einsum", "scipy")


@unittest.skipUnless(has_modules(*HUBBARD_DEPS), missing_message(*HUBBARD_DEPS))
class TestHubbardNumerics(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        add_src_to_path()
        import numpy as np

        from HubbardTweezer.Hubbard import Lattice, MLWF

        cls.np = np
        cls.Lattice = Lattice
        cls.MLWF = MLWF

    def build_chain(self):
        return self.Lattice(
            shape="square",
            lattice_symmetry=False,
            lattice=self.np.array([2]),
            lc=(1.5, 1.5),
        )

    def test_singleband_hubbard_is_well_formed(self):
        lattice = self.build_chain()
        mlwf = self.MLWF(
            N=6,
            lattice=lattice,
            R0=self.np.array([3.0, 0.0, 0.0]),
            dim=1,
            model="sho",
            trap=(1.0, 1.0),
            symmetry=False,
            verbosity=0,
            Nintgrl_grid=65,
        )
        A, U, WF = mlwf.singleband_Hubbard(u=True)
        self.assertEqual(A.shape, (2, 2))
        self.assertEqual(U.shape, (2,))
        self.assertTrue(self.np.allclose(A, A.conj().T, atol=1e-10))
        self.assertTrue(self.np.allclose(WF.conj().T @ WF, self.np.eye(2), atol=1e-10))
        self.assertTrue(self.np.all(U > 0))
        self.assertTrue(self.np.all(self.np.diff(mlwf.wf_centers[:, 0]) >= 0))
        self.assertTrue(self.np.allclose(U[0], U[1], rtol=0.2, atol=1e-6))
