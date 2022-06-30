from typing import ItemsView, Iterable
import numpy as np
import numpy.linalg as la


def simdiag(ops: Iterable[np.ndarray], evals=False, safe_mode=False) -> ItemsView[np.ndarray, np.ndarray]:
    N = ops[0].shape[0]

    val0, sol0 = la.eigh(ops[0])
    # nonzeroidx = np.nonzero(val0)[0]
    # zeroidx = np.logical_not(nonzeroidx)
    # sol0 = sol0[:, nonzeroidx]
    print(sol0)
    print(sol0.T @ ops[0] @ sol0)
    # R1 = np.zeros((N, N))
    if len(ops) == 2:
        # Theoretically, sol1 block diagonalizes ops[1]
        # But ops[0], ops[1] are not exactly commuting
        # THis algorithm is not stable
        # R1[nonzeroidx, :][:, nonzeroidx] = sol0.conj().T @ ops[1] @ sol0
        # R1[zeroidx, :][:, zeroidx] = sol0.conj().T @ ops[1] @ sol0
        R1 = sol0.conj().T @ ops[1] @ sol0
        print(f'R1 = {R1}')
        val1, sol1 = la.eigh(R1)
        # print(sol2.T @ R1 @ sol2)
        solution = sol0 @ sol1
        print(f'sol1 = {sol1}')
    else:
        solution = np.eye(N)
    print(solution.T @ ops[0] @ solution)
    print(solution.T @ ops[1] @ solution)
    return solution
