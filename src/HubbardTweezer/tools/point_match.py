# from ortools.linear_solver import pywraplp
from ortools.graph.python import linear_sum_assignment
from scipy.spatial.distance import cdist
import numpy as np


def nearest_match(site: np.ndarray, wf: np.ndarray) -> np.ndarray:
    # Match Wannier functions to nearest trap sites
    # site: (M, 2) array of trap site coordinates
    # wf: (N, 2) array of Wannier functions' centers of mass
    # A standard assignment problem, use Hungarian algorithm in O(N^3) time
    # https://en.wikipedia.org/wiki/Assignment_problem

    # i-th row is the distance of i-th site to each WFs
    dist_mat = cdist(site, wf, metric="euclidean")
    num_site = site.shape[0]
    num_wf = wf.shape[0]
    if num_site != num_wf:
        raise ValueError('Number of sites and WFs must be equal.')
    threshold = 0.5 * num_site

    end_nodes_unraveled, start_nodes_unraveled = np.meshgrid(
        np.arange(num_wf), np.arange(num_site)
    )
    start_nodes = start_nodes_unraveled.ravel()
    end_nodes = end_nodes_unraveled.ravel()
    arc_costs = dist_mat.ravel()
    
    assignment = linear_sum_assignment.SimpleLinearSumAssignment()
    assignment.add_arcs_with_cost(start_nodes, end_nodes, arc_costs)
    status = assignment.solve()
    
    order = np.arange(num_site)
    if status == assignment.OPTIMAL:
        # print(f"Total cost = {assignment.optimal_cost()}\n")
        for i in range(0, assignment.num_nodes()):
            # print(
            #     f"Worker {i} assigned to task {assignment.right_mate(i)}."
            #     + f"  Cost = {assignment.assignment_cost(i)}"
            # )
            order[i] = assignment.right_mate(i)
    elif status == assignment.INFEASIBLE:
        print("WARNING: No assignment is possible. Order is not changed.")
    elif status == assignment.POSSIBLE_OVERFLOW:
        print("Some input costs are too large and may cause an integer overflow.")
    
    if assignment.optimal_cost() > threshold:  # If any WF is way too far from its site
        print(f'WARNING: total distance = {assignment.optimal_cost()}.',
              'Assignment might be non-local.')

    # # Create the mip solver with the SCIP backend.
    # solver: pywraplp.Solver = pywraplp.Solver.CreateSolver('SCIP')

    # # match[i, j] is an array of 0-1 variables, which will be 1
    # # if site i is assigned to WF j.
    # match = {}
    # for i in range(num_site):
    #     for j in range(num_wf):
    #         match[i, j] = solver.IntVar(0, 1, '')

    # # Each site is assigned to exactly 1 WF.
    # for i in range(num_site):
    #     solver.Add(solver.Sum([match[i, j] for j in range(num_wf)]) == 1)

    # # Each WF is assigned to exactly 1 site.
    # for j in range(num_wf):
    #     solver.Add(solver.Sum([match[i, j] for i in range(num_site)]) == 1)

    # objective_terms = []
    # for i in range(num_site):
    #     for j in range(num_wf):
    #         objective_terms.append(dist_mat[i][j] * match[i, j])
    # solver.Minimize(solver.Sum(objective_terms))
    # status = solver.Solve()

    # order = np.arange(num_site)
    # if status == pywraplp.Solver.OPTIMAL or status == pywraplp.Solver.FEASIBLE:
    #     # print(f'Total cost = {solver.Objective().Value()}\n')
    #     for i in range(num_site):
    #         for j in range(num_wf):
    #             # Test if x[i,j] is 1 (with tolerance for floating point arithmetic).
    #             if match[i, j].solution_value() > 0.5:
    #                 # print(f'Site {i} assigned to WF {j}.' +
    #                 #       f' Dist: {dist_mat[i][j]}')
    #                 order[i] = j
    # else:
    #     print('WARNING: no solution found. Order is not changed.')
    # if solver.Objective().Value() > threshold:  # If any WF is way too far from its site
    #     print(f'WARNING: total distance = {solver.Objective().Value()}.',
    #           'Assignment might be non-local.')
    return order
