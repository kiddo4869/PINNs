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

def main(args: argparse.Namespace):

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    start = time.time()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("device:", device)

    x = np.linspace(0, 1, 100)
    y = np.linspace(0, 1, 100)
    plt.plot(x, y)
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

