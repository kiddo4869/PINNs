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

from data.training_data import training_data
from models.network import NN, PINN, UNET
from util.util import log_args

# Setting gloabl parameters
torch.set_default_dtype(torch.float32)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("device:", device)

def analytical_solution(x: Tensor, y: Tensor) -> Tensor:
    return torch.sin(2 * np.pi * x) * torch.sin(2 * np.pi * y)

def plot_solution(u_pred: Tensor, X_u_train: Tensor, u_train: Tensor, output_path):

    #Ground truth
    fig_1 = plt.figure(1, figsize=(18, 5))
    plt.subplot(1, 3, 1)
    plt.pcolor(x_1, x_2, usol, cmap='jet')
    plt.colorbar()
    plt.xlabel(r'$x_1$', fontsize=18)
    plt.ylabel(r'$x_2$', fontsize=18)
    plt.title('Ground Truth $u(x_1,x_2)$', fontsize=15)

    # Prediction
    plt.subplot(1, 3, 2)
    plt.pcolor(x_1, x_2, u_pred, cmap='jet')
    plt.colorbar()
    plt.xlabel(r'$x_1$', fontsize=18)
    plt.ylabel(r'$x_2$', fontsize=18)
    plt.title('Predicted $\hat u(x_1,x_2)$', fontsize=15)

    # Error
    plt.subplot(1, 3, 3)
    plt.pcolor(x_1, x_2, np.abs(usol - u_pred), cmap='jet')
    plt.colorbar()
    plt.xlabel(r'$x_1$', fontsize=18)
    plt.ylabel(r'$x_2$', fontsize=18)
    plt.title(r'Absolute error $|u(x_1,x_2)- \hat u(x_1,x_2)|$', fontsize=15)
    plt.tight_layout()

    plt.savefig(os.path.join(output_path, "Helmholtz_non_stiff.png"), dpi = 500, bbox_inches='tight')

def solutionplot(u_pred,X_u_train,u_train):

    #Ground truth
    fig_1 = plt.figure(1, figsize=(18, 5))
    plt.subplot(1, 3, 1)
    plt.pcolor(x_1, x_2, usol, cmap='jet')
    plt.colorbar()
    plt.xlabel(r'$x_1$', fontsize=18)
    plt.ylabel(r'$x_2$', fontsize=18)
    plt.title('Ground Truth $u(x_1,x_2)$', fontsize=15)

    # Prediction
    plt.subplot(1, 3, 2)
    plt.pcolor(x_1, x_2, u_pred, cmap='jet')
    plt.colorbar()
    plt.xlabel(r'$x_1$', fontsize=18)
    plt.ylabel(r'$x_2$', fontsize=18)
    plt.title('Predicted $\hat u(x_1,x_2)$', fontsize=15)

    # Error
    plt.subplot(1, 3, 3)
    plt.pcolor(x_1, x_2, np.abs(usol - u_pred), cmap='jet')
    plt.colorbar()
    plt.xlabel(r'$x_1$', fontsize=18)
    plt.ylabel(r'$x_2$', fontsize=18)
    plt.title(r'Absolute error $|u(x_1,x_2)- \hat u(x_1,x_2)|$', fontsize=15)
    plt.tight_layout()

    plt.savefig('Helmholtz_non_stiff.png', dpi = 500, bbox_inches='tight')

def main(args: argparse.Namespace):

    logger = logging.basicConfig(filename=os.path.join(args.log_path, "log"), level=logging.INFO)
    log_args(args)

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

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
    logging.info("\n----------model----------")
    logging.info(model)

    params = list(model.parameters())
    num_of_params = int(sum(p.numel() for p in model.parameters() if p.requires_grad))
    logging.info(f"Number of parameters: {num_of_params}")

    optimizer = torch.optim.Adam(model.parameters(),
                                 lr=args.learning_rate,
                                 betas=(0.9, 0.999),
                                 eps=1e-08,
                                 weight_decay=0,
                                 )
    logging.info(optimizer)

    # Training
    epochs = args.epochs

    start_time = time.time()

    # debugging

    for epoch in range(1, epochs + 1):

        train_loss = []
        valid_loss = []

        #for phase in ["train", "valid"]:

        optimizer.zero_grad()
        loss = model.loss(X_u_train, u_train)
        loss.backward()
        optimizer.step()
        
        if epoch % args.save_epoch_freq == 0:
            print(f"Epoch {epoch}/{epochs}, Loss: {loss.item():05f}")
            train_loss.append(loss.item())
    
    plt.imshow(phi, origin="lower", extent=[x_min, x_max, y_min, y_max])
    plt.savefig(os.path.join(args.output_path, "test.png"))

    end_time = time.time()
    print(f"training time elapsed: {(end_time - start_time):02f}s")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Physics Informed Neural Networks")
    
    # General parameters
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--model_path", type=str, default="./models")
    parser.add_argument("--output_path", type=str, default="./outputs")
    parser.add_argument("--log_path", type=str, default="./log")

    # Training parameters
    parser.add_argument("--learning_rate", type=float, default=1e-3)
    parser.add_argument("--epochs", type=int, default=1000)
    parser.add_argument("--save_epoch_freq", type=int, default=1)

    args = parser.parse_args()
    main(args)

