import torch.nn as nn
import torch
import numpy as np
import torchvision
from torchvision import transforms


def load_encoder(PATH='mnist_encoder.pt', device='cpu'):
    model = P_MNIST()
    model.load_state_dict(torch.load(PATH))
    model.eval()
    return model  

class P_MNIST(nn.Module):
    
    def __init__(self, z_dim = 100, nc = 1, ndf=16, ndir = 8, device = 'cpu'):
        super(P_MNIST,self).__init__()
        self.nc = nc
        self.ndf = ndf
        self.z_dim = z_dim
        self.device = device
        self.ndir = ndir
        self.layers = nn.Sequential(
            # (nc+1) x 28 x 28
            nn.Conv2d(nc, ndf, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            # (ndf) x 14 x 14
            nn.Conv2d(ndf, ndf*2, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            # (ndf*2) x 7 x 7
            nn.Conv2d(ndf*2, ndf*4, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            # (ndf*4) x 3 x 3
        )
        self.mu = nn.Sequential(
            nn.Linear(ndf*4*3*3, self.z_dim*self.ndir)
        )
        
    def forward(self, X):
        X = self.layers(X).view(-1, self.ndf * 4 * 3 * 3)
        return self.mu(X).chunk(self.ndir, dim=1)
