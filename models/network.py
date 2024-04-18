import torch
import torch.nn as nn
import numpy as np

class NN(nn.Module):
    def __init__(self):
        super(NN, self).__init__()
        self.fc1 = nn.Linear(1, 10)
        self.fc2 = nn.Linear(10, 1)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x

class PINN(nn.Module):
    def __init__(self, layers: list[int]):
        super(PINN, self).__init__()
        self.layers = layers

        # activation function
        self.activation = nn.Tanh()

        # loss function
        self.loss_function = nn.MSELoss(reduction = "mean")

        # fully connected layers
        self.linears = nn.ModuleList([nn.Linear(self.layers[i], self.layers[i+1]) for i in range(len(layers)-1)])

        """Xavier Normal Initialization"""
        for i in range(len(self.layers)-1):
            
            # weights from a normal distribution with 
            # Recommended gain value for tanh = 5/3?
            nn.init.xavier_normal_(self.linears[i].weight.data, gain=1.0)
            
            # set biases to zero
            nn.init.zeros_(self.linears[i].bias.data)

    def forward(self, x):

        if torch.is_tensor(x) == False:
            x = torch.tensor(x)
        
        device = x.get_device()

        lb = np.array([0, 0])  # lower bound
        ub = np.array([1, 1])

        u_b = torch.from_numpy(ub).float().to(device)
        l_b = torch.from_numpy(lb).float().to(device)

        # preprocessing input 
        x = (x - l_b) / (u_b - l_b) # feature scaling
        
        a = x.float()
        for i in range(len(self.layers)-2):
            z = self.linears[i](a)
            a = self.activation(z)

        return self.linears[-1](a)

    def loss_BC(self, x, y):
        return self.loss_function(self.forward(x), y)

    def loss_PDE(self, x):
        return 0

    def loss(self, x, y):
        return self.loss_BC(x, y) + self.loss_PDE(x)

    def closure(self):
        pass
    
    def predict(self):
        pass

class UNET(nn.Module):
    pass