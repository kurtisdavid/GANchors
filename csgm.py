import torch
import torchvision
import numpy as np
from torch.autograd import backward
from sklearn.neighbors.kde import KernelDensity
from pytorch_pretrained_biggan import truncated_noise_sample

i_se = lambda x,y: torch.sum(torch.sum(torch.nn.MSELoss(reduction='none')(x,y),dim=1),dim=1)
i_se_3d = lambda x,y: torch.sum(torch.sum(torch.sum(torch.nn.MSELoss(reduction='none')(x,y),dim=1),dim=1), dim=1)
se = torch.nn.MSELoss(reduction='none')

# KDE based sampling
class CSGM(torch.nn.Module):

    def __init__(self, target, filter, G, num_samples,
                 BS = 64, init_threshold = 1e-2, threshold=0.05,
                 bandwidth = 0.1, lr = 1e-2):
        super(CSGM, self).__init__() 
        self.target = torch.FloatTensor(target).cuda()
        self.A = torch.FloatTensor(filter).cuda()
        self.num_samples = num_samples
        self.G = G
        self.n_pixels = np.sum(filter)
        self.threshold = threshold
        self.init_threshold = init_threshold
        self.BS = BS
        # determine the points for KDE 
        self.z, self.init_samples, self.init_bg = reconstruct_batch(target, filter, self.n_pixels,
                    G, num_samples, threshold=init_threshold, lr = lr)
        self.Dz = KernelDensity(kernel='gaussian', bandwidth=bandwidth).fit(self.z.reshape(num_samples,100))

    def update_sampler(self, bandwidth):
        self.Dz = KernelDensity(kernel='gaussian', bandwidth=bandwidth).fit(self.z.reshape(-1,100)) 
    def sample(self, num_samples):
        count = 0
        z_samples = []
        gen_samples = []
        bg_samples = []
        while count < num_samples:
            Z = self.Dz.sample(self.BS)
            Z = torch.FloatTensor(Z).cuda().view(-1,100,1,1)    
            gen = self.G(Z).view(Z.shape[0],28,28)
            yhat = gen*self.A
            error = i_se(yhat, self.target.unsqueeze(0).repeat(Z.shape[0],1,1))/self.n_pixels

            Z = Z[error <= self.threshold]
            gen = gen[error <= self.threshold]
            end = min(Z.shape[0], num_samples - count)
            bg = gen[:end]*(1-self.A)
            z_samples.append(Z[:end].data.cpu().numpy())
            gen_samples.append(gen[:end].data.cpu().numpy())
            bg_samples.append(bg.data.cpu().numpy()) 
            count += end
        
        z_samples   = np.concatenate(z_samples, axis=0)
        gen_samples = np.concatenate(gen_samples, axis=0)
        bg_samples  = np.concatenate(bg_samples, axis=0)
        
        return z_samples, gen_samples, bg_samples 

def reconstruct_batch_ImageNet(target, filter, n_pixels, G, 
                               num_samples, z_dim=128, img_dim=128, n_channels = 3,
                               n_iter = 1000, threshold=0.05, truncation=1.0):
    y = torch.FloatTensor(target).cuda()
    A = torch.FloatTensor(filter).cuda()
    z = torch.FloatTensor(truncated_noise_sample(truncation=truncation, batch_size = 64)).cuda()
    z_param = torch.nn.Parameter(z)
    #batch_y = y.unsqueeze(0).repeat(z.shape[0],1,1)
    complete_zs = [] #np.array([]) #torch.nn.Parameter()

    lr = 1e-2
#    optim = torch.optim.SGD([z_param], lr=lr, momentum=0.9)
    optim = torch.optim.Adam([z_param], lr=lr)
    
    step = 0
    last_size = num_samples
    sampled = []
    while last_size > 0:
#        print(batch_y.shape)
        if (step > 1000) or z_param.shape[0] == 0:
            print('restarting with ',last_size, ' left')
            step = 0
            z = torch.FloatTensor(truncated_noise_sample(truncation=truncation, batch_size = 64)).cuda()
            z_param = torch.nn.Parameter(z)
#            optim = torch.optim.SGD([z_param], lr=lr, momentum=0.9)
            optim = torch.optim.Adam([z_param], lr=lr)
     #       batch_y = y.unsqueeze(0).repeat(z.shape[0],1,1)

        step += 1
        optim.zero_grad()
        x_hat = G(z_param, truncation).view(z_param.size()[0],n_channels,img_dim,img_dim)
        y_hat = x_hat * A

        loss = i_se_3d(y_hat,y)/(n_pixels*n_channels)
        loss_filt = loss[loss.data > threshold]
        loss_val = loss_filt.data.cpu().numpy()

        if loss_filt.shape[0] > 0:
            loss_mean = loss_filt.mean()
            loss_mean.backward()
            optim.step()
        z_completed = z_param[loss.data <= threshold]
        if z_completed.size()[0] != 0:
            remaining = last_size
            last_size -= z_completed.shape[0]
            min_len = min(remaining, z_completed.shape[0])
            sampled.append(x_hat[loss.data <= threshold].data.cpu().numpy()[:min_len])
            complete_zs.append(z_completed.data.cpu().numpy()[:min_len].reshape(min_len, z_dim, 1, 1))
#            z = z_param.data[loss.data > threshold]

            z[loss.data <= threshold] = torch.FloatTensor(truncated_noise_sample(truncation=truncation, batch_size = z_completed.shape[0])).cuda()
            # also cutoff those that have anything larger than truncation level...
            cutoffs = torch.sum ( torch.abs(z) > 1, dim = 1) > 0
            z[cutoffs] = torch.FloatTensor(truncated_noise_sample(truncation=truncation,
                                                                  batch_size=torch.sum(cutoffs).item())).cuda() 
            z_param = torch.nn.Parameter(z)
#            optim = torch.optim.SGD([z_param], lr=lr, momentum=0.9)
            optim = torch.optim.Adam([z_param], lr=lr)
            #if z_param.shape[0] > 0:
             #   batch_y = y.unsqueeze(0).repeat(z_param.shape[0],1,1)
        #optim = torch.optim.SGD([z_param], lr=1, momentum=0.9)
    
    complete_zs = np.concatenate(complete_zs, axis=0)
    final_sample = np.concatenate(sampled,axis=0)
    unmasked = torch.from_numpy(final_sample).cuda() * (1-A)

    return complete_zs, final_sample, unmasked.data.cpu().numpy()
    
    

def reconstruct_batch(target, filter, n_pixels, G,
                num_samples, z_dim=100, img_dim=28, n_channels=1, n_iter = 1000, threshold = 0.05, lr=1e-2, opt='adam', lambda_ = 0, def_size = 64):
    y = torch.FloatTensor(target).cuda()
    A = torch.FloatTensor(filter).cuda()
    z = torch.randn(def_size,z_dim,1,1,requires_grad = True).cuda()
    z_param = torch.nn.Parameter(z)
    batch_y = y.unsqueeze(0).repeat(z.shape[0],1,1)
    complete_zs = [] #np.array([]) #torch.nn.Parameter()
    # Repalce SGD with Adam?
    
    if opt == 'adam':
        optim = torch.optim.Adam([z_param], lr=lr)
    else:
        optim = torch.optim.SGD([z_param], lr=lr, momentum=0.9)
    
    step = 0
    last_size = num_samples
    sampled = []
    first_lr = lr 
    while last_size > 0:
#        print(batch_y.shape)
        if (step > 1000) or z_param.shape[0] == 0:
            print('restarting with ',last_size, ' left')
            step = 0
            z = torch.randn(def_size,z_dim,1,1,requires_grad = True).cuda()
            z_param = torch.nn.Parameter(z)
            if opt == 'adam':
                optim = torch.optim.Adam([z_param], lr=lr)
            else:
                optim = torch.optim.SGD([z_param], lr=lr, momentum=0.9)

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
            reg = z_param.norm(p=2) * lambda_
            loss_f = loss_mean + reg
            if step % 50 == 0:
                print(np.amin(loss_val))
            loss_f.backward()
            optim.step()
        if step % 400 == 0 and opt != 'adam':
            print("updating lr...")
            lr /= 10
            optim = torch.optim.SGD([z_param], lr=lr, momentum=0.9)
        z_completed = z_param[loss.data <= threshold]
        if z_completed.size()[0] != 0:
            remaining = last_size
            last_size -= z_completed.shape[0]
            min_len = min(remaining, z_completed.shape[0])
            sampled.append(x_hat[loss.data <= threshold].data.cpu().numpy()[:min_len])
            complete_zs.append(z_completed.data.cpu().numpy()[:min_len].reshape(min_len, z_dim, 1, 1))
#            z = z_param.data[loss.data > threshold]
            z[loss.data <= threshold] = torch.randn(z_completed.shape[0],100,1,1, requires_grad=True).cuda()
            z_param = torch.nn.Parameter(z)
            if opt == 'adam':
                optim = torch.optim.Adam([z_param], lr=lr)
            else:
                optim = torch.optim.SGD([z_param], lr=lr, momentum=0.9)
            if z_param.shape[0] > 0:
                batch_y = y.unsqueeze(0).repeat(z_param.shape[0],1,1)
        #optim = torch.optim.SGD([z_param], lr=1, momentum=0.9)
    
    complete_zs = np.concatenate(complete_zs, axis=0)
    final_sample = np.concatenate(sampled,axis=0)
    unmasked = torch.from_numpy(final_sample).cuda() * (1-A)

    return complete_zs, final_sample, unmasked.data.cpu().numpy()

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
