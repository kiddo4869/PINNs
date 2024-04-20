import os
import json
import argparse
from typing import List, Optional, Tuple, Dict, Union
import logging
import time
import numpy as np
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.optim as optim
from torch import Tensor

from data.training_data import training_data
from models.network import PINN, UNET
from util.util import log_args
from util.plotting import plot_solution

# Setting gloabl parameters
torch.set_default_dtype(torch.float32)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("device:", device)

def analytical_solution(X: Tensor, Y: Tensor) -> Tensor:
    return torch.sin(2 * torch.pi * X) * torch.sinh(2 * torch.pi * Y) / np.sinh(2 * np.pi) + torch.sin(2 * torch.pi * (1 - X)) * torch.sinh(2 * torch.pi * (1 - Y)) / np.sinh(2 * np.pi)
    #return X ** 2 + Y ** 2

def main(args: argparse.Namespace):

    logger = logging.basicConfig(filename=os.path.join(args.log_path, "Log.log"), level=logging.INFO)
    log_args(args)

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    # Generate training data 
    N = 100

    # the ratio is so important
    N_boundary = 120                     # Number of training data
    N_collocation = 600                  # Number of collocation points
    x_min, x_max = 0, 1
    y_min, y_max = 0, 1
    x = torch.linspace(x_min, x_max, N)
    y = torch.linspace(y_min, y_max, N)
    X, Y = torch.meshgrid(x, y, indexing="ij")
    phi = analytical_solution(X, Y)

    # Testing data
    X_u_test = np.hstack((X.reshape(-1, 1), Y.reshape(-1, 1)))  # shape = (N^2, 2)
    u_test = phi.reshape(-1, 1)                                 # shape = (N^2, 1)

    # Training data
    X_f_train_np_array, X_u_train_np_array, u_train_np_array, collocation_points = training_data(X, Y, phi, N_boundary, N_collocation)
    
    # plot X_f_train_np_array
    plt.figure(figsize=(6, 6))
    plt.scatter(X_u_train_np_array[:, 0], X_u_train_np_array[:, 1], c="b", s=0.5, marker="x", label="Boundary points")
    plt.scatter(collocation_points[:, 0], collocation_points[:, 1], c="r", s=0.5, marker="o", label="Collocation points")
    plt.savefig(os.path.join(args.output_path, "boundary_collocation_points.png"))

    # Convert numpy arrays to torch tensors and move them to the device
    X_f_train = torch.from_numpy(X_f_train_np_array).float().to(device)     # shape = (N_boundary + N_collocation, 2)
    X_u_train = torch.from_numpy(X_u_train_np_array).float().to(device)     # shape = (N_boundary, 2)
    u_train = torch.from_numpy(u_train_np_array).float().to(device)         # shape = (N_boundary, 1)
    X_u_test = torch.from_numpy(X_u_test).float().to(device)                # shape = (N^2, 2)
    u_test = u_test.to(device)                                              # shape = (N^2, 1)
    f_hat = torch.zeros(X_f_train.shape[0], 1).to(device)                   # shape = (N_boundary + N_collocation, 1)
    collocation_points = torch.from_numpy(collocation_points).float().to(device) # shape = (N_collocation, 2)

    layers = np.array(args.layers)

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
                                 weight_decay=0.0,
                                 )

    logging.info(optimizer)

    # Training
    epochs = args.epochs

    start_time = time.time()

    # debugging

    train_loss = []
    valid_loss = []
    epoch_list = []

    for epoch in range(1, epochs + 1):

        """
        for phase in ["train", "valid"]:

            if phase == "train":
                model.train()
            else:
                model.eval()
        """

        optimizer.zero_grad()
        loss = model.loss(X_u_train, u_train, X_f_train)
        loss.backward()
        optimizer.step()

        if epoch % args.log_epoch_freq == 0:
            print(f"Epoch {epoch}/{epochs}, Loss: {loss.item():.9f}")
            train_loss.append(loss.item())
            epoch_list.append(epoch)

        if epoch % args.save_epoch_freq == 0:
            x = X_u_test[:, 0].reshape(-1, 1)
            y = X_u_test[:, 1].reshape(-1, 1)
            u_pred = model(x, y)
            u_pred = u_pred.detach().cpu().numpy().reshape(N, N)
            plot_solution(epoch, u_pred, phi, X, Y, args.output_path)

        # after some epochs, we can reduce the learning rate
        if epoch == 20000:
            for param_group in optimizer.param_groups:
                param_group["lr"] = 0.0001

    end_time = time.time()
    print(f"training time elapsed: {(end_time - start_time):02f}s")

    # testing
    x = X_u_test[:, 0].reshape(-1, 1)
    y = X_u_test[:, 1].reshape(-1, 1)
    u_pred = model(x, y)
    u_pred = u_pred.detach().cpu().numpy().reshape(N, N)
    plot_solution(epoch, u_pred, phi, X, Y, args.output_path)

    with torch.no_grad():
        plt.figure(figsize=(10, 6))
        plt.plot(epoch_list, train_loss, label="train loss")
        plt.legend()
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.savefig(os.path.join(args.output_path, "losses.png"))

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Physics Informed Neural Networks")
    
    # General parameters
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--model_path", type=str, default="./models")
    parser.add_argument("--output_path", type=str, default="./outputs")
    parser.add_argument("--log_path", type=str, default="./log")
    parser.add_argument("--debug", action="store_true")

    # Training parameters
    parser.add_argument("--layers", type=json.loads, default=[2,20,20,20,20,1])
    parser.add_argument("--learning_rate", type=float, default=1e-2)
    parser.add_argument("--epochs", type=int, default=1000)
    parser.add_argument("--log_epoch_freq", type=int, default=100)
    parser.add_argument("--save_epoch_freq", type=int, default=1000)

    args = parser.parse_args()
    main(args)

