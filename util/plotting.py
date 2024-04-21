import os
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch import Tensor

def plot_boundary_collocation_points(boundary_points: np.array, collocation_points: np.array, output_path: str):
    plt.close()
    plt.figure(figsize=(6, 6))
    plt.title("Boundary and Collocation points")
    plt.xlabel("x")
    plt.ylabel("y")
    plt.scatter(boundary_points[:, 0], boundary_points[:, 1], c="b", s=2.5, marker="x", label="Boundary points")
    plt.scatter(collocation_points[:, 0], collocation_points[:, 1], c="r", s=2.5, marker="o", label="Collocation points")
    plt.legend(loc="upper right")
    plt.savefig(os.path.join(output_path, "boundary_collocation_points.png"))

def plot_solution(epoch: int, u_pred: Tensor, u_sol: Tensor, x: Tensor, y: Tensor, output_path: str):

    plt.close()

    #Ground truth
    fig_1 = plt.figure(1, figsize=(18, 5))
    plt.subplot(1, 3, 1)
    plt.pcolor(x, y, u_sol, cmap="jet")
    plt.colorbar()
    plt.xlabel(r'$x$', fontsize = 18)
    plt.ylabel(r'$y$', fontsize = 18)
    plt.title('Ground Truth $u(x, y)$', fontsize = 15)

    # Prediction
    plt.subplot(1, 3, 2)
    plt.pcolor(x, y, u_pred, cmap="jet")
    plt.colorbar()
    plt.xlabel(r'$x$', fontsize = 18)
    plt.ylabel(r'$y$', fontsize = 18)
    plt.title('Predicted $\hat u(x, y)$', fontsize = 15)

    # Error
    plt.subplot(1, 3, 3)
    plt.pcolor(x, y, np.abs(u_sol - u_pred), cmap = "jet")
    plt.colorbar()
    plt.xlabel(r'$x$', fontsize = 18)
    plt.ylabel(r'$y$', fontsize = 18)
    plt.title(r'Absolute error $|u(x, y) - \hat u(x, y)|$', fontsize = 15)
    plt.tight_layout()

    plt.savefig(os.path.join(output_path, f"prediction_epoch_{epoch}.png"), dpi = 100, bbox_inches = "tight")