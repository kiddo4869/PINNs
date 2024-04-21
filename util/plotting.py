import os
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch import Tensor
from PIL import Image

def plot_boundary_collocation_points(boundary_points: np.array, collocation_points: np.array, output_path: str, title: str):
    
    plt.close()

    plt.figure(figsize=(6, 6))
    plt.title(title)
    plt.xlabel("x")
    plt.ylabel("y")
    plt.scatter(boundary_points[:, 0], boundary_points[:, 1], c="b", s=2.5, marker="x", label="Boundary points")
    plt.scatter(collocation_points[:, 0], collocation_points[:, 1], c="r", s=2.5, marker="o", label="Collocation points")
    plt.legend(loc="upper right")
    name = title.replace(" ", "_").lower()
    plt.savefig(os.path.join(output_path, f"{name}.png"))

def plot_solution(epoch: int, u_pred: Tensor, u_sol: Tensor, x: Tensor, y: Tensor, loss: float, output_path: str):

    plt.close()

    # Ground truth
    fig = plt.figure(1, figsize=(18, 5.5))

    ax1 = plt.subplot(1, 3, 1)
    plt.pcolor(x, y, u_sol, cmap="jet")
    plt.colorbar()
    plt.xlabel(r"$x$", fontsize = 18)
    plt.ylabel(r"$y$", fontsize = 18)
    plt.title(r"Ground Truth $\phi(x, y)$", fontsize = 15)
    ax1.set_aspect("equal", adjustable = "box")

    # Prediction
    ax2 = plt.subplot(1, 3, 2)
    plt.pcolor(x, y, u_pred, cmap="jet")
    plt.colorbar()
    plt.xlabel(r"$x$", fontsize = 18)
    plt.ylabel(r"$y$", fontsize = 18)
    plt.title(r"Predicted $\hat \phi(x, y)$", fontsize = 15)
    ax2.set_aspect("equal", adjustable = "box")

    # Error
    ax3 = plt.subplot(1, 3, 3)
    plt.pcolor(x, y, np.abs(u_sol - u_pred), cmap = "jet")
    plt.colorbar()
    plt.xlabel(r"$x$", fontsize = 18)
    plt.ylabel(r"$y$", fontsize = 18)
    plt.title(r"Absolute Error $|\phi(x, y) - \hat \phi(x, y)|$", fontsize = 15)
    ax3.set_aspect("equal", adjustable = "box")

    plt.suptitle(f"\nTraining epoch: {epoch} | MSE: {loss:.5e}\n", fontsize = 15)

    plt.savefig(os.path.join(output_path, f"prediction_epoch_{epoch}.png"), dpi = 100, bbox_inches = "tight")

def save_gif_PIL(outfile, files, fps=10, loop=0):
    "Helper function for saving GIFs"
    imgs = [Image.open(file) for file in files]
    imgs[0].save(fp=outfile, format='GIF', append_images=imgs[1:], save_all=True, duration=int(1000/fps), loop=loop)