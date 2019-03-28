import torch
import torchvision
import numpy as np

se = torch.nn.MSELoss(reduction='none')
def reconstruct(target, filter, n_pixels, G, 
                z_dim=100, img_dim=28, n_channels=1, n_iter = 1000):
    z = torch.randn(1, z_dim, 1, 1, requires_grad=True).cuda()
    z_param = torch.nn.Parameter(z)
    optim = torch.optim.SGD([z_param], lr=0.5, momentum=0.9)

    y = torch.FloatTensor(target).cuda()
    A = torch.FloatTensor(filter).cuda()

    z_iter = []
    loss_iter = []
    for i in range(n_iter):
        optim.zero_grad()
        x_hat = G(z_param).squeeze()
        y_hat = A * x_hat
        
        loss = torch.sum(se(y_hat,y))/n_pixels
        loss_iter.append(loss.data.cpu().numpy())
        z_iter.append(z_param.data.cpu().numpy())        
        if i % 10 == 0:
            print(i, loss.data)
        loss.backward()
        optim.step()
    
    return z_iter[np.argmin(loss_iter,axis=0)]
