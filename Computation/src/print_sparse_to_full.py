from typing import Callable
import numpy as np
import numpy.linalg as la


def print_sparse_to_full(applyH: Callable[[np.ndarray], np.ndarray],
                         N: int) -> np.ndarray:
    identity = np.eye(N)
    Hmat = np.zeros((N, N))
    for i in range(N):
        Hmat[:, i] = applyH(identity[:, i])
    return Hmat
