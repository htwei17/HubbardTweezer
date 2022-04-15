import numpy as np
import numpy.linalg as la


def full(applyH, N: int) -> np.ndarray:
    identity = np.eye(N)
    Hmat = np.zeros((N, N))
    for i in range(N):
        Hmat[:, i] = applyH(identity[:, i])
    return Hmat