import torch
import torchvision
import numpy as np

se = torch.nn.MSELoss(reduction='none')
def reconstruct(target, filter, n_pixels, G, 
                num_samples, z_dim=100, img_dim=28, n_channels=1, n_iter = 100):
    z = torch.randn(num_samples, z_dim, 1, 1, requires_grad=True).cuda()
    z_param = torch.nn.Parameter(z)
    optim = torch.optim.SGD([z_param], lr=0.5, momentum=0.9)

    y = torch.FloatTensor(target).cuda()
    A = torch.FloatTensor(filter).cuda()

    for i in range(n_iter):
        optim.zero_grad()
        x_hat = G(z_param).squeeze()
        y_hat = x_hat * A
        
        loss = torch.sum(se(y_hat,y))/(n_pixels*num_samples)
        loss.backward()
        optim.step()
    sampled = G(z_param).squeeze()
    unmasked = sampled * (1-A) 
    return sampled.data.cpu().numpy(), unmasked.data.cpu().numpy()
