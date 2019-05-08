import torch
import torchvision
import numpy as np
from torch.autograd import backward
from sklearn.neighbors.kde import KernelDensity
from pytorch_pretrained_biggan import truncated_noise_sample
from pytorch_pretrained_biggan import (BigGAN, one_hot_from_names, truncated_noise_sample, save_as_images)
import torchvision.models as models

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

def reconstruct_celebA_batch(t, mask_tensor, n_pixels, G, device, bs_,
                               num_samples, z_dim=128, img_dim=128, n_channels = 3,
                               n_iter = 1000, threshold=0.05, truncation=1.0):
    bs = bs_
    noise1 = torch.randn(bs, 512).to(device)
    noise_param1 = torch.nn.Parameter(noise1)
    num_pixels = torch.sum(mask_tensor)
    print("Anchor takes: ", (num_pixels/(3*(256**2))).item()*100//1,"% of image pixels")
    opt = torch.optim.Adam(lr=1e-2, params=[noise_param1])
    batch_loss = 1
    lastloss=1
    itr_count = 0
    while batch_loss > threshold:
        itr_count += 1
        if (lastloss * 100) // 1 != (batch_loss * 100) // 1:
            print(itr_count, (batch_loss.item()*10000)//1 * .0001)
            print((loss1/(num_pixels/3)))
        lastloss=batch_loss
        sample_image = torch.nn.functional.interpolate(G(noise_param1, depth=8, alpha=1), scale_factor=1/4)
        masked_sample = mask_tensor*sample_image
        e = (masked_sample - t)
        se = e ** 2
        loss1 = torch.sum(torch.sum(torch.mean(se,dim=1),dim=-1),dim=-1)
        #print((loss1/(num_pixels/3)))
        batch_loss = loss1.sum()/(num_pixels*bs/3)
        #loss2 = torch.sum(se)/num_pixels
        #print(loss1.item() // .0001 ==  loss2.item() //.0001)
        batch_loss.backward()
        opt.step()

    return noise1, sample_image, masked_sample


def reconstruct_batch_celebA(target, filter, n_pixels, G,
                               num_samples, z_dim=128, img_dim=128, n_channels = 3,
                               n_iter = 1000, threshold=0.05, truncation=1.0):
    bs_ = 1
    G.to('cuda')
    class_vector = one_hot_from_names(['dog'], batch_size=bs_)
    class_vector = torch.from_numpy(class_vector)
    class_vec = class_vector.to('cuda')
    y = torch.FloatTensor(target).cuda()
    A = torch.FloatTensor(filter).cuda()
    z = torch.FloatTensor(truncated_noise_sample(truncation=truncation, batch_size = bs_)).cuda()
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
        if (step > 100000) or z_param.shape[0] == 0:
            print('restarting with ',last_size, ' left')
            step = 0
            z = torch.FloatTensor(truncated_noise_sample(truncation=truncation, batch_size = bs_)).cuda()
            z_param = torch.nn.Parameter(z)
#            optim = torch.optim.SGD([z_param], lr=lr, momentum=0.9)
            optim = torch.optim.Adam([z_param], lr=lr)
     #       batch_y = y.unsqueeze(0).repeat(z.shape[0],1,1)

        step += 1
        optim.zero_grad()
        x_hat = G(z_param, class_vec, truncation).view(z_param.size()[0],n_channels,img_dim,img_dim)
        y_hat = x_hat * A

        loss = i_se_3d(y_hat,y)/(n_pixels*n_channels)
        loss_filt = loss[loss.data > threshold]
        loss_val = loss_filt.data.cpu().numpy()
        print(loss_val)

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

def reconstruct_batch_threshold(target, filter, n_pixels, G,
                num_samples, threshold,  z_dim=100, img_dim=28, n_channels=1, n_iter = 1000, lr=1e-2, 
                opt='adam', lambda_ = 0, def_size = 64, init_mu = None):
    y = torch.FloatTensor(target).cuda()
    A = torch.FloatTensor(filter).cuda()
    # closure to allow initial predictions
    def create_z(def_size):
        if init_mu is None:
            z = torch.randn(def_size,z_dim,1,1,requires_grad = True).cuda()
        else:
            idx = np.random.randint(0,len(init_mu), size=(def_size,))
            z = torch.zeros(def_size, z_dim).cuda()
            for i in range(def_size):
                z[i,:] = init_mu[idx[i]].cuda() + torch.randn(1, z_dim).cuda()
            z = z.view(-1,z_dim,1,1) 
        return z
    z = create_z(def_size)
    z_param = torch.nn.Parameter(z)
    batch_y = y.unsqueeze(0).repeat(z.shape[0],1,1)
    complete_zs = []
    threshold_t = torch.zeros(max(def_size, threshold.shape[0])).cuda()
    threshold_t[:num_samples] = torch.FloatTensor(threshold).cuda()
    threshold = torch.sort(threshold_t,dim=-1,descending=True)[0].clone()
    del threshold_t
    threshold_current = threshold[:def_size].clone()
    current_threshold_size = def_size
    if opt == 'adam':
        optim = torch.optim.Adam([z_param], lr=lr)
    else:
        optim = torch.optim.SGD([z_param], lr=lr, momentum=0.9)
    thresholds_passed = []
    losses_passed = []
    step = 0
    last_size = num_samples
    sampled = []
    first_lr = lr
    while last_size > 0:
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

        loss = i_se(y_hat,batch_y)/(n_pixels+1e-9)
        # sort losses to match with largest thresholds for quickest collection
        loss, argloss = torch.sort(loss)
        # can we further optimize this by making sure smallest loss goes to smallest threshold greater than it?
        threshold_current = torch.sort(threshold_current, descending=True)[0]
        loss_filt = loss[loss.data > threshold_current.view(loss.shape)]
        loss_val = loss_filt.data.cpu().numpy()
#        if step % 50 == 0:
#            print(np.amin(loss_val))

        if loss_filt.shape[0] > 0:
            loss_mean = loss_filt.mean()
            reg = z_param.norm(p=2) * lambda_
            loss_f = loss_mean + reg
#            if step % 50 == 0:
 #               print(np.amin(loss_val))
            loss_f.backward()
            optim.step()
        if step % 400 == 0 and opt != 'adam':
            print("updating lr...")
            lr /= 10
            optim = torch.optim.SGD([z_param], lr=lr, momentum=0.9)
        correct_idx = argloss[loss.data <= threshold_current.view(loss.shape)]
        z_completed = z_param[correct_idx]
        if z_completed.size()[0] != 0:
            # adjust parameters 
            remaining = last_size
            last_size -= z_completed.shape[0]
            # collect the good samples
            min_len = min(remaining, z_completed.shape[0])
            sampled.append(x_hat[correct_idx].data.cpu().numpy()[:min_len])
            complete_zs.append(z_completed.data.cpu().numpy()[:min_len].reshape(min_len, z_dim, 1, 1))
             
            # update z's to search for more 
            z = z[argloss]
            z[loss.data <= threshold_current.view(loss.shape)] = create_z( z_completed.shape[0] )
            
            # update the thresholds from queue
            if num_samples >= def_size: 
                end_threshold = min(num_samples,current_threshold_size + z_completed.shape[0])
            else:
                end_threshold = min(def_size, current_threshold_size + z_completed.shape[0])
            threshold_current[ (loss.data <= threshold_current.view(loss.shape)).nonzero()[:end_threshold-current_threshold_size]] = threshold[current_threshold_size:end_threshold].view(-1,1)
            
#            thresholds_passed.append(threshold_current[ loss.data <= threshold_current.view(loss.shape)])
#            losses_passed.append(loss[loss.data <= threshold_current.view(loss.shape)])
            threshold_current[(loss.data <= threshold_current.view(loss.shape)).nonzero()[end_threshold-current_threshold_size:z_completed.shape[0]]] = 1e-9 # just set super low thresholds if the queue is empty
          #  print(threshold_current)
            current_threshold_size = end_threshold
            # setup new optimizer for everything
            z_param = torch.nn.Parameter(z)
            if opt == 'adam':
                optim = torch.optim.Adam([z_param], lr=lr)
            else:
                optim = torch.optim.SGD([z_param], lr=lr, momentum=0.9)
            if z_param.shape[0] > 0:
                batch_y = y.unsqueeze(0).repeat(z_param.shape[0],1,1)
    
    complete_zs = np.concatenate(complete_zs, axis=0)
    final_sample = np.concatenate(sampled,axis=0)
    unmasked = torch.from_numpy(final_sample).cuda() * (1-A)
#    print("Done")
    return complete_zs, final_sample, unmasked.data.cpu().numpy()

def reconstruct_batch(target, filter, n_pixels, G,
                num_samples, z_dim=100, img_dim=28, n_channels=1, n_iter = 1000, threshold = 0.05, lr=1e-2, opt='adam', lambda_ = 0, def_size = 64,
                init_mu = None):
    y = torch.FloatTensor(target).cuda()
    A = torch.FloatTensor(filter).cuda()
    def create_z(def_size):
        if init_mu is None:
            z = torch.randn(def_size,z_dim,1,1,requires_grad = True).cuda()
        else:
            idx = np.random.randint(0,len(init_mu), size=(def_size,))
            z = torch.zeros(def_size, z_dim).cuda()
            for i in range(def_size):
                z[i,:] = init_mu[idx[i]].cuda() + torch.randn(1, z_dim).cuda()
            z = z.view(-1,z_dim,1,1)
        return z
    z = create_z(def_size)
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
            z = create_z(def_size)
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

        loss = i_se(y_hat,batch_y)/(n_pixels + 1e-9)
        loss_filt = loss[loss.data > threshold]
        loss_val = loss_filt.data.cpu().numpy()

        if loss_filt.shape[0] > 0:
            loss_mean = loss_filt.mean()
            reg = z_param.norm(p=2) * lambda_
            loss_f = loss_mean + reg
    #        if step % 50 == 0:
    #            print(np.amin(loss_val))
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
            z[loss.data <= threshold] = create_z(z_completed.shape[0])
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
