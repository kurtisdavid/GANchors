import torch
import torchvision
import numpy as np
from torch.autograd import backward
#se = lambda x,y: torch.sum(torch.sum(torch.nn.MSELoss(reduction='none')(x,y),dim=1),dim=1)
se = torch.nn.MSELoss(reduction='sum')
def reconstruct(target, filter, n_pixels, G, 
                num_samples, z_dim=100, img_dim=28, n_channels=1, n_iter = 100, threshold = 0.1):
    y = torch.FloatTensor(target).cuda()
    A = torch.FloatTensor(filter).cuda()
    best_z = []
    i = 0  
    # each sample must be independent (not batch)
    #   this is because the gradient will not try to improve worst z if mean is low... 
    while i < num_samples:
        z = torch.randn(1,z_dim,1,1,requires_grad = True).cuda()
        z_param = torch.nn.Parameter(z)
        optim = torch.optim.SGD([z_param], lr=1, momentum=0.9)
        
        loss_val = 100000
        j = 0
        restart = False
        while loss_val > threshold:  
            optim.zero_grad()
            x_hat = G(z_param).squeeze()
            y_hat = x_hat * A
            loss = se(y_hat,y)/n_pixels
            loss_val = loss.data.cpu().numpy()
            loss.backward()
            optim.step()
            # random restart
            if j > 1000:
                restart = True
                break
            j += 1
        # try again...
        if restart:
            #print("restarting...")
            continue
        print(i,j,loss_val)
        i += 1
        best_z.append(z_param.data.cpu().numpy().reshape(z_dim,1,1))
    sampled = G(torch.from_numpy(np.array(best_z)).cuda()).view(-1,28,28)
    unmasked = sampled * (1-A)
    return sampled.data.cpu().numpy(), unmasked.data.cpu().numpy()
