from ortools.linear_solver import pywraplp
from scipy.spatial.distance import cdist
import numpy as np


def nearest_match(site: np.ndarray, wf: np.ndarray) -> np.ndarray:
    # Match Wannier functions to nearest trap sites
    # site: (M, 2) array of trap site coordinates
    # wf: (N, 2) array of Wannier functions' centers of mass

    # i-th row is the distance of i-th site to each WFs
    dist_mat = cdist(site, wf, metric="euclidean")
    num_site = site.shape[0]
    num_wf = wf.shape[0]
    if num_site != num_wf:
        raise ValueError('Number of sites and WFs must be equal.')

    # Create the mip solver with the SCIP backend.
    solver: pywraplp.Solver = pywraplp.Solver.CreateSolver('SCIP')

    # match[i, j] is an array of 0-1 variables, which will be 1
    # if site i is assigned to WF j.
    match = {}
    for i in range(num_site):
        for j in range(num_wf):
            match[i, j] = solver.IntVar(0, 1, '')

    # Each site is assigned to exactly 1 WF.
    for i in range(num_site):
        solver.Add(solver.Sum([match[i, j] for j in range(num_wf)]) == 1)

    # Each WF is assigned to exactly 1 site.
    for j in range(num_wf):
        solver.Add(solver.Sum([match[i, j] for i in range(num_site)]) == 1)

    objective_terms = []
    for i in range(num_site):
        for j in range(num_wf):
            objective_terms.append(dist_mat[i][j] * match[i, j])
    solver.Minimize(solver.Sum(objective_terms))
    status = solver.Solve()

    order = np.arange(num_site)
    if status == pywraplp.Solver.OPTIMAL or status == pywraplp.Solver.FEASIBLE:
        # print(f'Total cost = {solver.Objective().Value()}\n')
        for i in range(num_site):
            for j in range(num_wf):
                # Test if x[i,j] is 1 (with tolerance for floating point arithmetic).
                if match[i, j].solution_value() > 0.5:
                    # print(f'Site {i} assigned to WF {j}.' +
                    #       f' Dist: {dist_mat[i][j]}')
                    order[i] = j
    else:
        print('Warning: no solution found. Order is not changed.')
    return order
