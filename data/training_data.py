from typing import Tuple
import numpy as np
from torch import Tensor
from pyDOE import lhs               # Latin Hypercube Sampling

def training_data(X: Tensor, Y: Tensor, phi: Tensor, N_sample: int, N_collocation: int) -> Tuple[Tensor, Tensor, Tensor]:
    
    # boundary conditions
    leftedge_inputs = np.hstack((X[:, 0].reshape(-1, 1), Y[:, 0].reshape(-1, 1)))
    leftedge_phi = phi[:, 0].reshape(-1, 1)
    
    rightedge_inputs = np.hstack((X[:, -1].reshape(-1, 1), Y[:, -1].reshape(-1, 1)))
    rightedge_phi = phi[:, -1].reshape(-1, 1)
    
    topedge_inputs = np.hstack((X[0, :].reshape(-1, 1), Y[0, :].reshape(-1, 1)))
    topedge_phi = phi[0, :].reshape(-1, 1)
    
    bottomedge_inputs = np.hstack((X[-1, :].reshape(-1, 1), Y[-1, :].reshape(-1, 1)))
    bottomedge_phi = phi[-1, :].reshape(-1, 1)

    boundary_inputs = np.vstack([leftedge_inputs, rightedge_inputs, topedge_inputs, bottomedge_inputs])
    boundary_phi = np.vstack([leftedge_phi, rightedge_phi, topedge_phi, bottomedge_phi])

    # sample random points in the domain
    idx = np.random.choice(boundary_inputs.shape[0], N_sample, replace=False)

    sampled_boundary_inputs = boundary_inputs[idx, :]
    sampled_boundary_phi = boundary_phi[idx, :]

    # Domain bounds
    lb = np.array([0, 0])  # lower bound
    ub = np.array([1.0, 1.0])  # upper bound

    # Collocation points for training the model
    collocation_points = lb + (ub - lb) * lhs(2, N_collocation)                    # Latin Hypercube Sampling
    training_inputs = np.vstack((sampled_boundary_inputs, collocation_points))     # append the boundary points to the collocation points

    return training_inputs, sampled_boundary_inputs, sampled_boundary_phi, collocation_points