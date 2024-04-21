from typing import Tuple
import numpy as np
from torch import Tensor
from pyDOE import lhs               # Latin Hypercube Sampling

def data_processing(X: np.array, Y: np.array, phi: np.array, N_boundary: int, N_collocation: int, ub: np.array, lb: np.array) -> Tuple[np.array, np.array, np.array, np.array]:

    # Function to stack inputs and phi for edges
    def stack_edges(X_edge, Y_edge, phi_edge):
        inputs = np.hstack((X_edge.reshape(-1, 1), Y_edge.reshape(-1, 1)))
        phi_edge = phi_edge.reshape(-1, 1)
        return inputs, phi_edge

    # Boundary conditions
    leftedge_inputs, leftedge_phi = stack_edges(X[:, 0], Y[:, 0], phi[:, 0])
    rightedge_inputs, rightedge_phi = stack_edges(X[:, -1], Y[:, -1], phi[:, -1])
    topedge_inputs, topedge_phi = stack_edges(X[0, :], Y[0, :], phi[0, :])
    bottomedge_inputs, bottomedge_phi = stack_edges(X[-1, :], Y[-1, :], phi[-1, :])

    boundary_points = np.vstack([leftedge_inputs, rightedge_inputs, topedge_inputs, bottomedge_inputs])
    boundary_phi = np.vstack([leftedge_phi, rightedge_phi, topedge_phi, bottomedge_phi])

    # Sample random points in the boundary
    idx = np.random.choice(boundary_points.shape[0], N_boundary, replace=False)
    sampled_boundary_points = boundary_points[idx, :]
    sampled_boundary_phi = boundary_phi[idx, :]

    # Sample collocation points in the domain
    collocation_points = lb + (ub - lb) * lhs(2, N_collocation)                    # Latin Hypercube Sampling
    training_inputs = np.vstack((sampled_boundary_points, collocation_points))     # append the boundary points to the collocation points

    return training_inputs, sampled_boundary_points, sampled_boundary_phi, collocation_points