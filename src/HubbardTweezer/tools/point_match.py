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
    # Table dim 0: number of workers = number of sites
    # Table dim 1: number of tasks = number of WFs
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
            # print(f'Site {i} assigned to WF {assignment.right_mate(i)}.' +
            #               f' Dist: {assignment.assignment_cost(i)}')
            order[i] = assignment.right_mate(i)
    elif status == assignment.INFEASIBLE:
        print("WARNING: No assignment is possible. Order is not changed.")
    elif status == assignment.POSSIBLE_OVERFLOW:
        print("Some input costs are too large and may cause an integer overflow.")
    
    if assignment.optimal_cost() > threshold:  # If any WF is way too far from its site
        print(f'WARNING: total distance = {assignment.optimal_cost()}.',
              'Assignment might be non-local.')
    return order
