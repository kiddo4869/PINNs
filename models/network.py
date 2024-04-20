import torch
import torch.nn as nn
import torch.autograd as autograd
import numpy as np

class PINN(nn.Module):
    def __init__(self, layers: list[int]):
        super(PINN, self).__init__()
        self.layers = layers

        # activation function
        # if you use 2nd order derivatives for ReLU activations, you should have all zeros
        self.activation = nn.Tanh()

        # loss function
        self.loss_function = nn.MSELoss(reduction = "mean")

        # fully connected layers
        self.linears = nn.ModuleList([nn.Linear(self.layers[i], self.layers[i+1]) for i in range(len(layers)-1)])

        # batch normalization layers
        self.batch_norms = nn.ModuleList([nn.BatchNorm1d(self.layers[i+1]) for i in range(len(layers)-1)])

        # dropout layer
        self.dropout = nn.Dropout(p=0.0)

        for i in range(len(self.layers)-1):
            
            # weights from a normal distribution with 
            # Recommended gain value for tanh = 5/3?
            nn.init.xavier_normal_(self.linears[i].weight.data, gain=1.0)
            
            # set biases to zero
            nn.init.zeros_(self.linears[i].bias.data)

    def forward(self, x, y):

        u = torch.cat((x, y), dim = 1)
        
        device = u.get_device()

        lb = np.array([0, 0])  # lower bound
        ub = np.array([1, 1])

        u_b = torch.from_numpy(ub).float().to(device)
        l_b = torch.from_numpy(lb).float().to(device)

        # preprocessing input 
        u = (u - l_b) / (u_b - l_b) # feature scaling
        
        a = u.float()
        for i in range(len(self.layers)-2):
            z = self.linears[i](a)
            #z = self.batch_norms[i](z)
            a = self.activation(z)
            #a = self.dropout(a)

        return self.linears[-1](a)

    def loss_BC(self, x, y, phi):
        return self.loss_function(self.forward(x, y), phi)

    def loss_PDE(self, x_collocation, y_collocation):

        x = x_collocation.clone().detach()
        y = y_collocation.clone().detach()

        x.requires_grad = True
        y.requires_grad = True

        phi = self.forward(x, y)

        device = x_collocation.get_device()
        
        phi_x = autograd.grad(phi, x,
                            create_graph = True,
                            grad_outputs = torch.ones_like(phi).to(device),
                            allow_unused = True)[0]
        phi_y = autograd.grad(phi, y,
                            create_graph = True,
                            grad_outputs = torch.ones_like(phi).to(device),
                            allow_unused = True)[0]
        phi_xx = autograd.grad(phi_x, x,
                            create_graph = True,
                            grad_outputs = torch.ones_like(phi).to(device),
                            allow_unused = True)[0]
        phi_yy = autograd.grad(phi_y, y,
                            create_graph = True,
                            grad_outputs = torch.ones_like(phi).to(device),
                            allow_unused = True)[0]

        f = phi_xx + phi_yy

        loss_f = self.loss_function(f, torch.zeros(f.shape).to(device))
        return loss_f

    def loss(self, X_u_train, u_train, X_f_train):
        x = X_u_train[:, 0].reshape(-1, 1)
        y = X_u_train[:, 1].reshape(-1, 1)
        phi = u_train
        x_collocation = X_f_train[:, 0].reshape(-1, 1)
        y_collocation = X_f_train[:, 1].reshape(-1, 1)

        weight = 0.5
        return self.loss_BC(x, y, phi) + weight * self.loss_PDE(x_collocation, y_collocation)

    def closure(self):
        pass
    
    def predict(self):
        pass

class UNET(nn.Module):
    pass