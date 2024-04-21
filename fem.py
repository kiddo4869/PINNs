import os
import argparse
import logging
from typing import List, Optional, Tuple, Dict, Union
import time

import torch
from torch import Tensor
import numpy as np
import matplotlib.pyplot as plt
from scipy import sparse
from scipy.sparse.linalg import eigsh
from scipy.sparse.linalg import eigs
from util.util import log_args
from util.plotting import plot_solution, plot_boundary_collocation_points, save_gif_PIL

def set_2d_boundaries(boundary_array: Tensor, x: Tensor, y: Tensor) -> Tensor:
    
    boundary_array[:, 0] = 0                            # Left Boundary
    boundary_array[:, -1] = 0                           # Right Boundary
    boundary_array[0, :] = - torch.sin(2 * np.pi * x)   # Bottom Boundary
    boundary_array[-1, :] = torch.sin(2 * np.pi * x)    # Top Boundary

    return boundary_array

def construct_matrix_A(N: int) -> np.array:
    
    diag = np.ones([N])
    diags = np.array([-diag, 2 * diag, -diag])
    D = sparse.spdiags(diags, np.array([-1, 0, 1]), N, N)
    A = sparse.kronsum(D, D).toarray()

    return A

def plot_matrix(A, Ainv, output_path: str):

    plt.close()

    fig = plt.figure(figsize = (12, 4))
    plt.subplot(121)
    plt.imshow(A, interpolation = "none")
    clb=plt.colorbar()
    clb.set_label("Matrix elements values")
    plt.title("Matrix A")
    plt.subplot(122)
    plt.imshow(Ainv, interpolation = "none")
    clb=plt.colorbar()
    clb.set_label("Matrix elements values")
    plt.title(r"Matrix $A^{-1}$")
    plt.savefig(os.path.join(output_path, "Matrix_A_Ainv.png"))

def analytical_solution(X: Tensor, Y: Tensor) -> Tensor:
    return torch.sin(2 * torch.pi * X) * torch.sinh(2 * torch.pi * Y) / np.sinh(2 * np.pi) + torch.sin(2 * torch.pi * (1 - X)) * torch.sinh(2 * torch.pi * (1 - Y)) / np.sinh(2 * np.pi)

def main(args: argparse.Namespace):

    os.makedirs(args.output_path, exist_ok=True)

    if args.debug:
        args.grid_size = 10

    # Generate grid size
    N = args.grid_size
    x_min, x_max = 0, 1
    y_min, y_max = 0, 1
    ub = np.array([x_max, y_max])
    lb = np.array([x_min, y_min])
    x = torch.linspace(x_min, x_max, N)
    y = torch.linspace(y_min, y_max, N)
    X, Y = torch.meshgrid(x, y, indexing="xy")
    phi = analytical_solution(X, Y)

    os.makedirs(args.output_path, exist_ok=True)

    logger = logging.basicConfig(filename=os.path.join(args.output_path, "Log.log"), level=logging.INFO)
    log_args(args)

    start_time = time.time()

    # Constructing the boundary conditions
    b = set_2d_boundaries(np.zeros((N, N)), x, y)
    b_column = b.T.reshape(-1, 1)

    logging.info(f"Convert {b.shape} into {b_column.shape}")

    # Constructing matrix A using scipy
    diag = np.ones([N])
    diags = np.array([-diag, 2*diag, -diag])
    D = sparse.spdiags(diags, np.array([-1, 0, 1]), N, N)

    logging.info(f"\nMatrix D:\n{D.toarray()} \nwith the shape {D.toarray().shape}")

    # Kronecker sum of two sparse matrices
    A = construct_matrix_A(N)
    Ainv = np.linalg.inv(A)

    logging.info(f"\nMatrix A:\n{A} \nwith the shape {A.shape}")

    # Plotting the two matrix A and inverse matrix A^-1
    plot_matrix(A, Ainv, args.output_path)

    # obtain the column vector by dot product
    phi_column = np.dot(Ainv, b_column)

    # reshape the column vector into 2D grid
    phi_fem = phi_column.reshape((N, N)).T

    end_time = time.time()
    print(f"time elapsed: {(end_time - start_time):02f}s")
    logging.info(f"time elapsed: {(end_time - start_time):02f}s")

    # Plot the numerical solution
    plt.close()

    # Ground truth
    fig = plt.figure(1, figsize=(18, 5.5))

    ax1 = plt.subplot(1, 3, 1)
    plt.pcolor(x, y, phi, cmap="jet")
    plt.colorbar()
    plt.xlabel(r"$x$", fontsize = 18)
    plt.ylabel(r"$y$", fontsize = 18)
    plt.title(r"Ground Truth $\phi(x, y)$", fontsize = 15)
    ax1.set_aspect("equal", adjustable = "box")

    # Prediction
    ax2 = plt.subplot(1, 3, 2)
    plt.pcolor(x, y, phi_fem, cmap="jet")
    plt.colorbar()
    plt.xlabel(r"$x$", fontsize = 18)
    plt.ylabel(r"$y$", fontsize = 18)
    plt.title(r"Predicted $\hat \phi(x, y)$", fontsize = 15)
    ax2.set_aspect("equal", adjustable = "box")

    # Error
    ax3 = plt.subplot(1, 3, 3)
    plt.pcolor(x, y, np.abs(phi - phi_fem), cmap = "jet")
    plt.colorbar()
    plt.xlabel(r"$x$", fontsize = 18)
    plt.ylabel(r"$y$", fontsize = 18)
    plt.title(r"Absolute Error $|\phi(x, y) - \hat \phi(x, y)|$", fontsize = 15)
    ax3.set_aspect("equal", adjustable = "box")

    loss_function = torch.nn.MSELoss()
    loss = loss_function(torch.tensor(phi_fem), torch.tensor(phi)).item()

    plt.suptitle(f"\nNumerical Solution (Finite Element Method) | MSE: {loss:.5e}\n", fontsize = 15)

    plt.savefig(os.path.join(args.output_path, "fem_solution.png"))

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Physics Informed Neural Networks")
    
    # General parameters
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output_path", type=str, default="./outputs")
    parser.add_argument("--debug", action="store_true")
    parser.add_argument("--grid_size", type=int, default=100)

    args = parser.parse_args()
    main(args)

