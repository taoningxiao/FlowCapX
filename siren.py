import torch
from torch import nn
import numpy as np

class SIREN(nn.Module):
    def __init__(self, in_dim, out_dim, hidden_layers, hidden_dim):
        super(SIREN, self).__init__()

        self.configs = {
            'in_dim': in_dim,
            'out_dim': out_dim,
            'hidden_layers': hidden_layers,
            'hidden_dim': hidden_dim
        }
        self.layers = nn.ModuleList()
        self.omega0 = 30
        self.hidden_omega = 1
        for i in range(hidden_layers):
            # if i == 0:
            #     in_features = in_dim
            #     out_features = 64
            # elif i == 1:
            #     in_features = 64
            #     out_features = hidden_dim
            # elif i == hidden_layers - 1:
            #     in_features = hidden_dim
            #     out_features = out_dim
            # else:
            #     in_features = hidden_dim
            #     out_features = hidden_dim
            in_features = in_dim if i == 0 else hidden_dim
            out_features = out_dim if i == hidden_layers-1 else hidden_dim            
            self.layers.append(nn.Linear(in_features, out_features))
    
    def init_weight(self):
        in_dim = self.configs['in_dim']
        out_dim = self.configs['out_dim']
        hidden_layers = self.configs['hidden_layers']
        hidden_dim = self.configs['hidden_dim']
        with torch.no_grad():
            for i, layer in enumerate(self.layers):
                in_features = in_dim if i == 0 else hidden_dim
                out_features = out_dim if i == hidden_layers-1 else hidden_dim   
                if i == 0:
                    layer.weight.uniform_(-np.sqrt(6 / in_features) / self.omega0, np.sqrt(6 / in_features) / self.omega0)
                else:
                    layer.weight.uniform_(-np.sqrt(6 / in_features) / self.hidden_omega, np.sqrt(6 / in_features) / self.hidden_omega)
            

    def forward(self, x):
        # x_clone = x.clone()
        # x_clone[..., -1] = x_clone[..., -1] / 5
        # x = x_clone
        for i, layer in enumerate(self.layers):
            x = layer(x)
            if i == 0:
                x = torch.sin(self.omega0 * x)
            elif i < len(self.layers)- 1:
                x = torch.sin(self.hidden_omega * x)
        return x
        # x, y, z, t = x[0], x[1], x[2], x[3]
        # u = x * y + torch.sin(z)
        # v = y * z + torch.cos(t)
        # w = t * x + torch.exp(y)
        # return torch.stack([u, v, w])

    def grad(self, x):
        jacobian = torch.func.vmap(torch.func.jacfwd(self.forward), in_dims=0, out_dims=0)(x).squeeze()
        return jacobian
    
    def hessian(self, x):
        hessian = torch.func.vmap(torch.func.jacfwd(torch.func.jacfwd(self.forward)))(x).squeeze()
        return hessian


class SIREN_VEL(nn.Module):
    def __init__(self, in_dim, out_dim, hidden_layers, hidden_dim):
        super(SIREN_VEL, self).__init__()

        self.configs = {
            'in_dim': in_dim,
            'out_dim': out_dim,
            'hidden_layers': hidden_layers,
            'hidden_dim': hidden_dim
        }
        self.layers = nn.ModuleList()
        self.omega0 = 30
        self.hidden_omega = 1
        for i in range(hidden_layers):
            if i == 0:
                in_features = in_dim
                out_features = 64
            elif i == 1:
                in_features = 64
                out_features = hidden_dim
            elif i == hidden_layers - 1:
                in_features = hidden_dim
                out_features = out_dim
            else:
                in_features = hidden_dim
                out_features = hidden_dim
            # in_features = in_dim if i == 0 else hidden_dim
            # out_features = out_dim if i == hidden_layers-1 else hidden_dim            
            self.layers.append(nn.Linear(in_features, out_features))
            

    def forward(self, x):
        # x_clone = x.clone()
        # x_clone[..., -1] = x_clone[..., -1] / 5
        # x = x_clone
        for i, layer in enumerate(self.layers):
            x = layer(x)
            if i == 0:
                x = torch.sin(self.omega0 * x)
            elif i < len(self.layers)- 1:
                x = torch.sin(self.hidden_omega * x)
        return x
        # x, y, z, t = x[0], x[1], x[2], x[3]
        # u = x * y + torch.sin(z)
        # v = y * z + torch.cos(t)
        # w = t * x + torch.exp(y)
        # return torch.stack([u, v, w])

    def grad(self, x):
        jacobian = torch.func.vmap(torch.func.jacfwd(self.forward), in_dims=0, out_dims=0)(x).squeeze()
        return jacobian
    
    def hessian(self, x):
        hessian = torch.func.vmap(torch.func.jacfwd(torch.func.jacfwd(self.forward)))(x).squeeze()
        return hessian


def save_SIREN_model(model, path):
    checkpoint = {
        'model_state_dict': model.state_dict(),
        'config': model.configs
    }
    torch.save(checkpoint, path)


def load_SIREN_model(path, device='cuda'):
    checkpoint = torch.load(path, map_location=device)
    config = checkpoint['config']

    model = SIREN(
        in_dim=config['in_dim'],
        out_dim=config['out_dim'],
        hidden_layers=config['hidden_layers'],
        hidden_dim=config['hidden_dim']
    )
    model.load_state_dict(checkpoint['model_state_dict'])
    return model


if __name__ == "__main__":
    inputs = torch.tensor([
        [1.0, 2.0, 3.0, 4.0],  # 样本 1
        [0.5, 1.5, 2.5, 3.5],  # 样本 2
    ], requires_grad=True)
    model = SIREN(4, 3, 1, 1)
    jac = model.grad(inputs)
    print("jac.shape: ", jac.shape)
    print(jac)

    hes = model.hessian(inputs)
    print("hes.shape: ", hes.shape)
    print(hes)