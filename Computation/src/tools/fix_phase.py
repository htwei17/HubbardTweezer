from typing import Iterable
from numbers import Number
import numpy as np


def fix_phase(psi: np.ndarray) -> np.ndarray:
    if isinstance(psi, Iterable):
        psi = np.array(psi)
    elif isinstance(psi, Number):
        psi = abs(np.array([psi]))
        
    if psi.shape[0] > 0:
        if psi.ndim == 1:
            m = np.argmax(abs(psi))
            eta = np.conj(psi[m]) / abs(psi[m])

            if abs(psi[m]) < 0.001:
                print("Positify warning: small maodule!")
            return psi * eta
        else:
            nalpha = psi.shape[1]
            psi1 = psi.copy()
            for alpha in range(nalpha):
                psi1[:, alpha] = fix_phase(psi[:, alpha])
            return psi1
    else:
        print("Positify warning: empty array input!")
    return psi
