import torch
import torchvision
import numpy as np
from torch.autograd import backward
i_se = lambda x,y: torch.sum(torch.sum(torch.nn.MSELoss(reduction='none')(x,y),dim=1),dim=1)
se = torch.nn.MSELoss(reduction='none')
def reconstruct_batch(target, filter, n_pixels, G,
                num_samples, z_dim=100, img_dim=28, n_channels=1, n_iter = 1000, threshold = 0.05):
    y = torch.FloatTensor(target).cuda()
    A = torch.FloatTensor(filter).cuda()
    z = torch.randn(64,z_dim,1,1,requires_grad = True).cuda()
    z_param = torch.nn.Parameter(z)
    batch_y = y.unsqueeze(0).repeat(z.shape[0],1,1)
    complete_zs = [] #np.array([]) #torch.nn.Parameter()
    # Repalce SGD with Adam?
    #optim = torch.optim.SGD([z_param], lr=1, momentum=0.9)
    optim = torch.optim.Adam([z_param], lr=1)
    #for i in range(n_iter):
    step = 0
    last_size = num_samples
    while last_size > 0:
#        print(batch_y.shape)
        if (step > 1000):
            print('restarting with ',z_param.size()[0], ' left')
            step = 0
            z = torch.randn(64,z_dim,1,1,requires_grad = True).cuda()
            z_param = torch.nn.Parameter(z)
            optim = torch.optim.Adam([z_param], lr=1)
            batch_y = y.unsqueeze(0).repeat(z.shape[0],1,1)

        step += 1
        optim.zero_grad()
        x_hat = G(z_param).view(z_param.size()[0],28,28)
        y_hat = x_hat * A

        loss = i_se(y_hat,batch_y)/(n_pixels)
        loss_filt = loss[loss.data > threshold]
        loss_val = loss_filt.data.cpu().numpy()

        if loss_filt.shape[0] > 0:
            loss_mean = loss_filt.mean()
            loss_mean.backward()
            optim.step()

        z_completed = z_param[loss.data < threshold]
        if z_completed.size()[0] != 0:
            remaining = last_size
            for i in range(z_completed.size()[0]):
                last_size -= 1
            min_len = min(remaining, z_completed.size()[0])
            complete_zs.append(z_completed.data.cpu().numpy()[:min_len].reshape(min_len, z_dim, 1, 1))
            z_param = torch.nn.Parameter(z_param[loss.data > threshold])
            optim = torch.optim.Adam([z_param], lr=1)
            if z_param.shape[0] > 0:
                batch_y = y.unsqueeze(0).repeat(z_param.shape[0],1,1)
        #optim = torch.optim.SGD([z_param], lr=1, momentum=0.9)

    final_z = np.array(complete_zs[0])
    for arr in complete_zs[1:]:
       final_z = np.concatenate((final_z, arr), axis=0)

    sampled = G(torch.from_numpy(np.array(final_z)).cuda()).view(num_samples,28,28)
    unmasked = sampled * (1-A)

    return sampled.data.cpu().numpy(), unmasked.data.cpu().numpy()
'''
    sampled = G(z_param).view(-1,28,28)
    losses = i_se(sampled,y)/n_pixels
    unmasked = sampled * (1-A)
    return sampled.data.cpu().numpy(), unmasked.data.cpu().numpy(),losses.data.cpu().numpy()
'''
def reconstruct(target, filter, n_pixels, G,
                num_samples, z_dim=100, img_dim=28, n_channels=1, n_iter = 100, threshold = 0.2):
    y = torch.FloatTensor(target).cuda()
    A = torch.FloatTensor(filter).cuda()
    best_z = []
    i = 0
    # each sample must be independent (not batch)
    #   this is because the gradient will not try to improve worst z if mean is low...
    while i < num_samples:
        z = torch.randn(1,z_dim,1,1,requires_grad = True).cuda()
        z_param = torch.nn.Parameter(z)
        optim = torch.optim.Adam([z_param], lr=1)

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
            #if j:
            #    print(j,loss_val)
            # random restart
            if j > 1000:
            #    restart = True
                break
            j += 1
        # try again...
        if restart:
            print("restarting...", loss_val)
            continue
        print(i,j,loss_val)
        i += 1
        best_z.append(z_param.data.cpu().numpy().reshape(z_dim,1,1))
    sampled = G(torch.from_numpy(np.array(best_z)).cuda()).view(-1,28,28)
    unmasked = sampled * (1-A)
    return sampled.data.cpu().numpy(), unmasked.data.cpu().numpy()
