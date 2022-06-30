"""
Routines for simultaneous diagonalization
Hao-Tian Wei <weihaotian776@gmail.com?
"""

import numpy as np
import numpy.linalg as la


def givens_rotate(A: np.ndarray, i, j, c, s) -> np.ndarray:
    # Rotate A along axis (i,j) by c and s
    R = np.array([[c, s], [-s, c]])
    A[[i, j], :] = R @ A[[i, j], :]
    return A


def givens_double_rotate(A: np.ndarray, i, j, c, s) -> np.ndarray:
    # Double rotate A along axis (i,j) by c and s
    R = np.array([[c, s], [-s, c]])
    A[[i, j], :] = R @ A[[i, j], :]
    A[:, [i, j]] = A[:, [i, j]] @ R.T
    return A


def jacobi_angles(*Ms, **kwargs):
    r"""
    Simultaneously diagonalize using Jacobi angles.

    Input:
        - Ms: list of matrices to diagonalize
        - kwargs: optional arguments
        
    Output:
        - R :
        - L :
        - err :

    @article{SC-siam,
       HTML =   "ftp://sig.enst.fr/pub/jfc/Papers/siam_note.ps.gz",
       author = "Jean-Fran\c{c}ois Cardoso and Antoine Souloumiac",
       journal = "{SIAM} J. Mat. Anal. Appl.",
       title = "Jacobi angles for simultaneous diagonalization",
       pages = "161--164",
       volume = "17",
       number = "1",
       month = jan,
       year = {1995}}

    (a) Compute Givens rotations for every pair of indices (i,j) i < j
        - from eigenvectors of G = gg'; g = A_ij - A_ji, A_ij + A_ji
        - Compute c, s as \sqrt{x+r/2r}, y/\sqrt{2r(x+r)}

    (b) Update matrices by multiplying by the givens rotation R(i,j,c,s)

    (c) Repeat (a) until stopping criterion: sin theta < threshold for all ij pairs
    """

    assert len(Ms) > 0
    m, n = Ms[0].shape
    assert m == n

    sweeps = kwargs.get('sweeps', 500)
    threshold = kwargs.get('eps', 1e-8)
    rank = kwargs.get('rank', m)

    R = np.eye(m)

    for _ in range(sweeps):
        done = True
        for i in range(rank):
            for j in range(i+1, m):
                G = np.zeros((2, 2))
                for M in Ms:
                    g = np.array([M[i, i] - M[j, j], M[i, j] + M[j, i]])
                    G += np.outer(g, g) / len(Ms)
                # Compute the eigenvector directly
                t_on, t_off = G[0, 0] - G[1, 1], G[0, 1] + G[1, 0]
                theta = 0.5 * np.arctan2(t_off, t_on +
                                         np.sqrt(t_on*t_on + t_off * t_off))
                c, s = np.cos(theta), np.sin(theta)

                if abs(s) > threshold:
                    done = False
                    # Update the matrices and V
                    for M in Ms:
                        givens_double_rotate(M, i, j, c, s)
                        #assert M[i,i] > M[j, j]
                    R = givens_rotate(R, i, j, c, s)

        if done:
            break
    R = R.T

    L = np.zeros((m, len(Ms)))
    err = 0
    for i, M in enumerate(Ms):
        # The off-diagonal elements of M should be 0
        L[:, i] = np.diag(M)
        err += la.norm(M - np.diag(np.diag(M)))

    return R, L, err
