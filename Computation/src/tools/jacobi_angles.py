"""
Jacobi angle method for simultaneous diagonalization.
Arun Chaganty <arunchaganty@gmail.com>
Hao-Tian Wei <weihaotian776@gmail.com?
"""
__all__ = ['jacobi_angles']

import imp
import numpy as np
import numpy.linalg as la

from .fix_phase import fix_phase


def givens_rotate(A: np.ndarray, i, j, R) -> np.ndarray:
    # Rotate A along axis (i,j) by c and s
    A[[i, j], :] = R @ A[[i, j], :]
    return A


def givens_double_rotate(A: np.ndarray, i, j, R) -> np.ndarray:
    # Double rotate A along axis (i,j) by c and s
    A[[i, j], :] = R @ A[[i, j], :]
    A[:, [i, j]] = A[:, [i, j]] @ R.conj().T
    return A


def update_rotation(Ms, eigvec, i, j, c, s):
    # Ms is immutable, so no need to output it back
    R = np.array([[c, s.conj()], [-s, c.conj()]])
    # print(f'R = {R}')

    # err = 0
    # tot = 0
    # Update the matrices and V
    for M in Ms:
        givens_double_rotate(M, i, j, R)
    #     err += la.norm(M - np.diag(np.diag(M)), 'fro')
    #     tot += la.norm(M, 'fro')
    # print(f'abs err = {err}')
    # print(f'rel err = {err / tot}')
    eigvec = givens_rotate(eigvec, i, j, R)
    return eigvec


def jacobi_angles(*Ms, **kwargs) -> tuple[np.ndarray, np.ndarray, float]:
    """
    Simultaneously diagonalize real normal matrices using Jacobi angles.

    Parameters
    ----------
        - *Ms = M1, M2, ... : sequance of matrices to diagonalize
        - kwargs: optional arguments

    Returns
    ----------
        - eigvec : list of eigenvectors
        - eigval : eigenvalues
        - err : error from the diagonalization

    Algorithm
    ----------
    (a) Compute Givens rotations for every pair of indices (i,j) i < j
        - from eigenvectors of G = \sum_k gg'; g = A_ij - A_ji, A_ij + A_ji
        - Compute c, s as \sqrt{x+r/2r}, y/\sqrt{2r(x+r)}

    (b) Update matrices by multiplying by the givens rotation R(i,j,c,s)

    (c) Repeat (a) until stopping criterion: sin theta < threshold for all ij pairs

    References
    ----------
    @article{JacobiAngles,

       HTML =   "ftp://sig.enst.fr/pub/jfc/Papers/siam_note.ps.gz",

       author = "Jean-Fran\c{c}ois Cardoso and Antoine Souloumiac",

       journal = "{SIAM} J. Mat. Anal. Appl.",

       title = "Jacobi angles for simultaneous diagonalization",

       pages = "161--164",

       volume = "17",

       number = "1",

       month = jan,

       year = {1995}}
    """

    Ms = list(Ms)
    assert len(Ms) > 0, "No matrices to diagonalize!"
    m, n = Ms[0].shape
    assert m == n, "Only square matrices can be diagonalized!"

    if len(Ms) == 1:
        print('Only one matrix given. Directly diagonalize.')
        eigval, eigvec = la.eigh(Ms[0])
        err = la.norm(Ms[0] - np.diag(np.diag(Ms[0])), 'fro')
        return eigvec, eigval[:, None], err

    sweeps = kwargs.get('sweeps', 500)
    threshold = kwargs.get('eps', 1e-8)
    rank = kwargs.get('rank', m)

    real = True
    eigvec = np.eye(m)

    wherecomplex = np.array(
        list(Ms[i].dtype == np.complex128 for i in range(len(Ms))))
    if wherecomplex.any():
        real = False
        wherereal = np.where(wherecomplex == False)[0]
        for idx in wherereal:
            Ms[idx] = Ms[idx].astype(np.complex128)
        eigvec = np.eye(m, dtype=np.complex128)

    for sweep in range(sweeps):
        done = True
        for i in range(rank):
            for j in range(i+1, m):
                if real:
                    G = np.zeros((2, 2), dtype=float)
                    for M in Ms:
                        g = np.array(
                            [M[i, i] - M[j, j], M[i, j] + M[j, i]])
                        G += np.outer(g, g)
                    # a = G[0, 0]
                    # b = G[0, 1]
                    # d = G[1, 1]
                    # if b == 0:
                    #     idx = np.argmax([a, d])
                    #     v = np.zeros(2)
                    #     v[idx] = 1
                    # else:
                    #     t = np.sqrt(a**2+4 * b**2-2 * a * d+d ** 2)
                    #     v = np.array([(a-d+t)/(2*b), 1])
                    #     v = v / la.norm(v)
                else:
                    G = np.zeros((3, 3), dtype=np.complex128)
                    for M in Ms:
                        g = np.array(
                            [M[i, i] - M[j, j], M[i, j] + M[j, i], 1j * (M[j, i] - M[i, j])])
                        G += np.outer(g, g)
                # Compute the eigenvector directly
                # print(f'G = {G}')
                e, w = la.eigh(G)
                v = fix_phase(w[:, np.argmax(e)])
                # print(f'e = {e}')
                # print(f'w = {w}')
                c = np.sqrt(0.5 + v[0]/2)
                s = v[1]
                if not real:
                    s -= 1j * v[2]
                s /= 2 * c

                print(f'Sweep No. = {sweep}\tRotation s = {abs(s)}')
                # s is rotation component in R, s==0 algo converges
                if abs(s) > threshold:
                    done = False
                    eigvec = update_rotation(Ms, eigvec, i, j, c, s)

        if done:
            print('Jacobi angle converged.')
            break
        if sweep == sweeps-1:
            print('Jacobi angle reaches max iter time.')

    eigvec = eigvec.conj().T

    eigval = np.zeros((m, len(Ms)))
    err = 0
    for i, M in enumerate(Ms):
        # The off-diagonal elements of M should be 0
        eigval[:, i] = np.real(np.diag(M))
        err += la.norm(M - np.diag(np.diag(M)), 'fro')

    return eigvec, eigval, err
