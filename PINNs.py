import os
import argparse
from typing import List, Optional, Tuple, Dict, Union
import logging
import time
import numpy as np
import matplotlib.pyplot as plt

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

def training_data(N: int) -> Tuple[Tensor, Tensor, Tensor]:
    pass

def main(args: argparse.Namespace):

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    start = time.time()

    N_u = 100    # Number of data points for the initial condition
    N_f = 10000  # Number of data points in the domain

    # Generate training data 
    N = 100
    x_min, x_max = 0, 1
    y_min, y_max = 0, 1
    x = torch.linspace(0, 1, N)
    y = torch.linspace(0, 1, N)
    X, Y = torch.meshgrid(x, y)
    phi = analytical_solution(X, Y).reshape(-1, 1)

    # boundary conditions
    leftedge_x = np.hstack((X[:, 0].reshape(-1, 1), Y[:, 0].reshape(-1, 1)))
    leftedge_x = np.hstack((X[:, 0].reshape(-1, 1), Y[:, 0].reshape(-1, 1)))
    leftedge_x = np.hstack((X[:, 0].reshape(-1, 1), Y[:, 0].reshape(-1, 1)))
    leftedge_x = np.hstack((X[:, 0].reshape(-1, 1), Y[:, 0].reshape(-1, 1)))
    
    print(leftedge_x)
    leftedge_u = phi[:,0][:,None]
    
    print(leftedge_x.shape)
    print(leftedge_u.shape)
    exit()
    # Domain bounds
    lb = np.array([-1, -1]) #lower bound
    ub = np.array([1, 1])  #upper bound

    a_1 = 1 
    a_2 = 1

    k = 1

    usol = np.sin(a_1 * np.pi * X) * np.sin(a_2 * np.pi * Y) #solution chosen for convinience  

    u_true = usol.flatten('F')[:,None] 
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

