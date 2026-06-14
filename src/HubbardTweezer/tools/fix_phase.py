"""
Fix the phase of each vector in the array.
Zhiyuan Wang
Hao-Tian Wei <weihaotian776@gmail.com?
"""
from typing import Iterable
from numbers import Number
import numpy as np


def fix_phase(psi: np.ndarray, mode: str = 'mvsd') -> np.ndarray:
    # Fix the phase of each vector in the array.
    # mode: 'mvsd' for multidiple vectors, each vector has 1 dimension.
    #       'svmd' for single vector with multiple dimensions.
    # FIXME: cannot do multiple vectors with multiple dimensions.
    if isinstance(psi, Iterable):
        psi = np.array(psi)
    elif isinstance(psi, Number):
        psi = abs(np.array([psi]))

    if psi.size == 0:
        raise ValueError("fix_phase: empty array input!")

    if 'mv' in mode or 'sd' in mode:
        return _fix_phase_multivec(psi)
    elif 'sv' in mode or 'md' in mode:
        return _fix_phase_multidim(psi)
    else:
        raise ValueError("fix_phase: invalid mode!")


def _fix_phase_multidim(psi: np.ndarray) -> np.ndarray:
    # Fix the phase of a single vector reshaped to multiple-dimension array.
    ps = psi.shape
    psi = psi.reshape(-1)
    m = np.argmax(abs(psi))
    eta = np.conj(psi[m]) / abs(psi[m])

    if abs(psi[m]) < 0.001:
        print("fix_phase warning: small maodule!")
    return psi.reshape(ps) * eta


def _fix_phase_multivec(psi: np.ndarray) -> np.ndarray:
    # Fix the phase of each vector in the array.
    ps = psi.shape
    if ps[0] > 0:
        if psi.ndim == 1:
            m = np.argmax(abs(psi))
            eta = np.conj(psi[m]) / abs(psi[m])

            if abs(psi[m]) < 0.001:
                print("fix_phase warning: small maodule!")
            return psi * eta
        else:
            nalpha = ps[1]
            psi1 = psi.copy()
            for alpha in range(nalpha):
                psi1[:, alpha] = _fix_phase_multivec(psi[:, alpha])
            return psi1
    else:
        print("fix_phase warning: empty array input!")
    return psi
