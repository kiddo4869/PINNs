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

from data.data_processing import data_processing
from models.network import PINN, UNET
from util.util import log_args
from util.plotting import plot_solution, plot_boundary_collocation_points, save_gif_PIL

# Setting gloabl parameters
torch.set_default_dtype(torch.float32)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("device:", device)

def analytical_solution(X: Tensor, Y: Tensor) -> Tensor:
    return torch.sin(2 * torch.pi * X) * torch.sinh(2 * torch.pi * Y) / np.sinh(2 * np.pi) + torch.sin(2 * torch.pi * (1 - X)) * torch.sinh(2 * torch.pi * (1 - Y)) / np.sinh(2 * np.pi)
    #return X ** 2 + Y ** 2

def main(args: argparse.Namespace):

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    # Generate grid size
    N = 100

    # the ratio is so important
    N_boundary = 120                     # Number of training data
    N_collocation = 600                  # Number of training collocation points
    N_boundary_val = 100                 # Number of validation data
    N_collocation_val = 100              # Number of validation collocation points

    # debugging
    if args.debug:
        N = 50
        N_boundary = 50
        N_collocation = 100
        N_boundary_val = 10
        N_collocation_val = 10
        args.epochs = 1000
        args.log_epoch_freq = 10
        args.save_epoch_freq = 10
        args.output_path = "./outputs/debug"

    os.makedirs(args.output_path, exist_ok=True)
    os.makedirs(os.path.join(args.output_path, "snapshots"), exist_ok=True)

    logger = logging.basicConfig(filename=os.path.join(args.output_path, "Log.log"), level=logging.INFO)
    log_args(args)

    x_min, x_max = 0, 1
    y_min, y_max = 0, 1
    ub = np.array([x_max, y_max])
    lb = np.array([x_min, y_min])
    x = torch.linspace(x_min, x_max, N)
    y = torch.linspace(y_min, y_max, N)
    X, Y = torch.meshgrid(x, y, indexing="ij")
    phi = analytical_solution(X, Y)

    # Add noise
    if args.noise_level > 0:
        noise = args.noise_level * torch.std(phi) * torch.randn_like(phi)
        phi_noisy = phi + noise

    # Training and validation data
    if args.noise_level > 0:
        training_inputs, training_bps, training_bphi, training_cps = data_processing(X, Y, phi_noisy, N_boundary, N_collocation, ub, lb)
        validation_inputs, validation_bps, validation_bphi, validation_cps = data_processing(X, Y, phi_noisy, N_boundary_val, N_collocation_val, ub, lb)
    else:
        training_inputs, training_bps, training_bphi, training_cps = data_processing(X, Y, phi, N_boundary, N_collocation, ub, lb)
        validation_inputs, validation_bps, validation_bphi, validation_cps = data_processing(X, Y, phi, N_boundary_val, N_collocation_val, ub, lb)
    
    # plot boundary and collocation points
    plot_boundary_collocation_points(training_bps, training_cps, args.output_path, "Training Boundary and Collocation Points")
    plot_boundary_collocation_points(validation_bps, validation_cps, args.output_path, "Validation Boundary and Collocation Points")

    # Convert numpy arrays to torch tensors and move them to the device
    X_u_train = torch.from_numpy(training_bps).float().to(device)
    u_train = torch.from_numpy(training_bphi).float().to(device)
    X_f_train = torch.from_numpy(training_inputs).float().to(device)

    X_u_val = torch.from_numpy(validation_bps).float().to(device)
    u_val = torch.from_numpy(validation_bphi).float().to(device)
    X_f_val = torch.from_numpy(validation_inputs).float().to(device)

    X_flatten = X.reshape(-1, 1).to(device)
    Y_flatten = Y.reshape(-1, 1).to(device)

    phi_gpu = phi.to(device)

    # Model
    model = PINN(np.array(args.layers), args.pinn)
    model.to(device)
    logging.info("\n----------model----------")
    logging.info(model)

    params = list(model.parameters())
    num_of_params = int(sum(p.numel() for p in model.parameters() if p.requires_grad))
    logging.info(f"Number of parameters: {num_of_params}")

    # Optimizer
    optimizer = torch.optim.Adam(model.parameters(),
                                 lr=args.learning_rate,
                                 betas=(0.9, 0.999),
                                 eps=1e-08,
                                 weight_decay=0.0,
                                 )

    logging.info(optimizer)

    # Training and validation
    epochs = args.epochs
    start_time = time.time()

    best_loss = np.inf
    epochs_no_improve = 0
    loss_function = nn.MSELoss(reduction="mean")

    train_loss = []
    valid_loss = []
    epoch_list = []
    files = []

    for epoch in range(1, epochs + 1):

        if epoch % args.log_epoch_freq == 0:
            epoch_list.append(epoch)

        # Training phase
        model.train()
        optimizer.zero_grad()
        loss_train = model.loss(X_u_train, u_train, X_f_train)
        loss_train.backward()
        optimizer.step()

        if args.log_loss:
            # Validation phase
            if epoch % args.log_epoch_freq == 0:
                model.eval()
                loss_val = model.loss(X_u_val, u_val, X_f_val)

                print(f"Epoch {epoch}/{epochs}, train loss: {loss_train.item():.9f}, valid loss: {loss_val.item():.9f}")
                train_loss.append(loss_train.item())
                valid_loss.append(loss_val.item())

                # Early stopping
                if args.early_stopping:
                    current_loss = loss_val.item()
                    if current_loss < best_loss - args.tolerance:
                        best_loss = current_loss
                        epochs_no_improve = 0
                    else:
                        epochs_no_improve += 1
                        if epochs_no_improve == args.patience:
                            print(f"Early stopping at epoch {epoch}")
                            break

        if args.log_sol:
            # Save the solution
            if epoch % args.save_epoch_freq == 0:
                with torch.no_grad():
                    u_pred = model(X_flatten, Y_flatten).reshape(N, N)
                    test_loss = loss_function(u_pred, phi_gpu).item()
                    
                    u_pred = u_pred.detach().cpu().numpy()
                    plot_solution(epoch, u_pred, phi, X, Y, test_loss, os.path.join(args.output_path, "snapshots"))
                    file = os.path.join(args.output_path, "snapshots", f"prediction_epoch_{epoch}.png")
                    files.append(file)

        # after some epochs, we can reduce the learning rate
        if epoch == 40000:
            for param_group in optimizer.param_groups:
                param_group["lr"] = 0.0001

        if epoch == 60000:
            for param_group in optimizer.param_groups:
                param_group["lr"] = 0.00001

    end_time = time.time()
    print(f"training time elapsed: {(end_time - start_time):02f}s")
    logging.info(f"training time elapsed: {(end_time - start_time):02f}s")

    if args.log_sol:
        save_gif_PIL(os.path.join(args.output_path, "prediction.gif"), files)

    # testing
    u_pred = model(X_flatten, Y_flatten).reshape(N, N)
    test_loss = loss_function(u_pred, phi_gpu).item()
    u_pred = u_pred.detach().cpu().numpy()
    plot_solution(epoch, u_pred, phi, X, Y, test_loss, args.output_path)

    if args.log_loss:
        with torch.no_grad():
            plt.figure(figsize=(10, 6))
            plt.plot(epoch_list, train_loss, label="train loss")
            plt.plot(epoch_list, valid_loss, label="valid loss")
            plt.legend(fontsize=15)
            plt.xlabel("Epoch", fontsize=15)
            plt.ylabel("Losses", fontsize=15)
            plt.savefig(os.path.join(args.output_path, "losses.png"))

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Physics Informed Neural Networks")
    
    # General parameters
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--model_path", type=str, default="./models")
    parser.add_argument("--output_path", type=str, default="./outputs")
    parser.add_argument("--debug", action="store_true")
    parser.add_argument("--pinn", action="store_true")
    parser.add_argument("--log_loss", action="store_true")
    parser.add_argument("--log_sol", action="store_true")
    parser.add_argument("--noise_level", type=float, default=0.0)

    # Training parameters
    parser.add_argument("--layers", type=json.loads, default=[2,20,20,20,20,1])
    parser.add_argument("--learning_rate", type=float, default=1e-2)
    parser.add_argument("--epochs", type=int, default=1000)
    parser.add_argument("--log_epoch_freq", type=int, default=100)
    parser.add_argument("--save_epoch_freq", type=int, default=1000)
    parser.add_argument("--early_stopping", action="store_true")
    parser.add_argument("--patience", type=int, default=10)
    parser.add_argument("--tolerance", type=float, default=1e-5)

    args = parser.parse_args()
    main(args)

