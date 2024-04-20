import os
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch import Tensor

def plot_solution(epoch: int, u_pred: Tensor, u_sol: Tensor, x: Tensor, y: Tensor, output_path):

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