import numpy as np
import torch
import pymanopt


eps = np.finfo(float).eps


def cost_func(U: torch.Tensor, R: list) -> torch.Tensor:
    # Cost function to Wannier optimize
    o = 0
    for Ri in R:
        # R is real-symmetric if no absorber
        X = U.conj().T @ Ri @ U
        Xp = X - torch.diag(torch.diag(X))
        o += torch.trace(torch.matrix_power(Xp, 2))
    # X must be hermitian, so is Xp
    # Xp^2 is then positive and hermitian,
    # its diagonal is real and positive, o >= 0
    # Min is found when X diagonal, which means U diagonalize R
    # SO U can be pure real orthogonal matrix!
    # Q: Can X, Y, Z be diagonalized simultaneously in high dims?
    # A: If the space is conplete then by QM theory it is possible
    #    to diagonalize X, Y, Z simultaneously.
    #    But this is not the case as it's incomplete.
    return o.real


def riemann_minimize(R: list[np.ndarray], x0=None, verbosity: int = 0) -> np.ndarray:
    # It can be proven that U can be purely real
    # See details in the paper & notes
    verbosity = int(np.clip(verbosity, 0, 3))
    N: int = R[0].shape[0]  # matrix dimension
    # Convert list of ndarray to list of Tensor
    R = [torch.from_numpy(Ri) for Ri in R]

    manifold = pymanopt.manifolds.SpecialOrthogonalGroup(N)

    @pymanopt.function.pytorch(manifold)
    def _cost_func(point: torch.Tensor) -> torch.Tensor:
        return cost_func(point, R)

    problem = pymanopt.Problem(manifold=manifold, cost=_cost_func)
    # By RMP 84.4(2012), cc is efficient
    # Cost func is always positive but not quadratic in U
    optimizer = pymanopt.optimizers.ConjugateGradient(
        max_iterations=1000,
        min_step_size=eps,
        min_gradient_norm=eps,
        verbosity=verbosity,
    )
    result = optimizer.run(problem, initial_point=x0, reuse_line_searcher=True)
    solution = result.point
    return solution
