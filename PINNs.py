import os
import argparse
from typing import List, Optional, Tuple, Dict, Union
import logging
import time
import numpy as np
import matplotlib.pyplot as plt
from pyDOE import lhs               # Latin Hypercube Sampling

import torch
import torch.nn as nn
import torch.autograd as autograd
import torch.optim as optim
from torch import Tensor

from models.network import NN, PINN, UNET

# Setting gloabl parameters
torch.set_default_dtype(torch.float32)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("device:", device)

def analytical_solution(x: Tensor, y: Tensor) -> Tensor:
    return torch.sin(2 * np.pi * x) * torch.sin(2 * np.pi * y)

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
    ub = np.array([1, 1])  # upper bound

    # Collocation points for training the model
    collocation_points = lb + (ub - lb) * lhs(2, N_collocation)                    # Latin Hypercube Sampling
    training_inputs = np.vstack((sampled_boundary_inputs, collocation_points))     # append the boundary points to the collocation points

    return training_inputs, sampled_boundary_inputs, sampled_boundary_phi

def main(args: argparse.Namespace):

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    start = time.time()

    N_u = 100    # Number of data points for the initial condition
    N_f = 10000  # Number of data points in the domain

    # Generate training data 
    N = 100
    N_sample = 50
    N_collocation = 20
    x_min, x_max = 0, 1
    y_min, y_max = 0, 1
    x = torch.linspace(0, 1, N)
    y = torch.linspace(0, 1, N)
    X, Y = torch.meshgrid(x, y)
    phi = analytical_solution(X, Y)

    # Testing data
    X_u_test = np.hstack((X.reshape(-1, 1), Y.reshape(-1, 1)))
    u_test = phi.reshape(-1, 1)

    # Training data
    X_f_train_np_array, X_u_train_np_array, u_train_np_array = training_data(X, Y, phi, N_sample, N_collocation)
    
    # Convert numpy arrays to torch tensors and move them to the device
    X_f_train = torch.from_numpy(X_f_train_np_array).float().to(device)
    X_u_train = torch.from_numpy(X_u_train_np_array).float().to(device)
    u_train = torch.from_numpy(u_train_np_array).float().to(device)
    X_u_test = torch.from_numpy(X_u_test).float().to(device)
    u_test = u_test.to(device)
    f_hat = torch.zeros(X_f_train.shape[0], 1).to(device)

    layers = np.array([2, 50, 50, 50, 1])

    model = PINN(layers)
    model.to(device)
    print(model)

    params = list(model.parameters())
    print("Number of parameters: ", sum(p.numel() for p in model.parameters() if p.requires_grad))

    exit()
    
    plt.imshow(phi, origin="lower", extent=[x_min, x_max, y_min, y_max])
    plt.savefig(os.path.join(args.output_path, "test.png"))


    end = time.time()
    print(f"time taken elapsed: {(end - start):02f}s")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Physics Informed Neural Networks")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--model_path", type=str, default="./models")
    parser.add_argument("--output_path", type=str, default="./outputs")
    parser.add_argument("--log_path", type=str, default="log path")

    args = parser.parse_args()
    main(args)

