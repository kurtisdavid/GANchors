# originally from: https://github.com/csinva/pytorch_gan_pretrained/blob/master/mnist_dcgan/dcgan.py
# https://raw.githubusercontent.com/pytorch/examples/master/dcgan/main.py

from __future__ import print_function
import argparse
import os
import random
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torch.utils.data
import torchvision.datasets as dset
import torchvision.transforms as transforms
import torchvision.utils as vutils
import torch.nn.functional as F
# python dcgan.py --dataset mnist --dataroot /scratch/users/vision/yu_dl/raaz.rsk/data/cifar10 --imageSize 28 --cuda --outf . --manualSeed 13 --niter 100

i_se = lambda x,y: torch.sum(torch.sum(torch.nn.MSELoss(reduction='none')(x,y),dim=1),dim=1)

# container class for a GAN that also provides a log_P method
class ProbGenerator(nn.Module):
    
    def __init__(self, G, anchor, target, threshold = 0.05):
        super(ProbGenerator, self).__init__()
        self.G = G
        self.anchor = torch.FloatTensor(anchor).cuda() # just a binary filter
        self.num_pixels = self.anchor.sum()
        self.target = torch.FloatTensor(target).cuda() # filtered anchor
        self.threshold = threshold
 
    def forward(self, Z):
        return self.G(Z)
    
    def log_prob(self, Z, debug = False):
        gen = self.G(Z.view(Z.shape[0],100,1,1)).view(Z.shape[0],28,28)
        yhat = gen * self.anchor
        if debug:
            print(yhat.shape)
            print(self.num_pixels)
        error = i_se(yhat, self.target.unsqueeze(0).repeat(Z.shape[0],1,1))/self.num_pixels
        # if P = exp( -error ) -> log_P = -error
        return -error

    def filter_samples(self, Z):
        try:
            Z = torch.from_numpy(Z).type(torch.FloatTensor).cuda().view(-1,100,1,1)
        except:
            pass
        gen = self.G(Z.view(-1,100,1,1)).view(Z.shape[0],28,28)
        yhat = gen * self.anchor
        error = i_se(yhat, self.target.unsqueeze(0).repeat(Z.shape[0],1,1))/self.num_pixels
        return gen[error.data < self.threshold] 

            
        

def load_generator(path='./weights'):
    G = Generator(1).eval()
    G.load_state_dict(torch.load('./weights/netG_epoch_99.pth'))
    return G

class Generator(nn.Module):
    def __init__(self, ngpu, nc=1, nz=100, ngf=64):
        super(Generator, self).__init__()
        self.ngpu = ngpu
        self.main = nn.Sequential(
            # input is Z, going into a convolution
            nn.ConvTranspose2d(     nz, ngf * 8, 4, 1, 0, bias=False),
            nn.BatchNorm2d(ngf * 8),
            nn.ReLU(True),
            # state size. (ngf*8) x 4 x 4
            nn.ConvTranspose2d(ngf * 8, ngf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 4),
            nn.ReLU(True),
            # state size. (ngf*4) x 8 x 8
            nn.ConvTranspose2d(ngf * 4, ngf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 2),
            nn.ReLU(True),
            # state size. (ngf*2) x 16 x 16
            nn.ConvTranspose2d(ngf * 2,     ngf, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf),
            nn.ReLU(True),
            nn.ConvTranspose2d(    ngf,      nc, kernel_size=1, stride=1, padding=2, bias=False),
            nn.Tanh()
        )

    def forward(self, input):
        if input.is_cuda and self.ngpu > 1:
            output = nn.parallel.data_parallel(self.main, input, range(self.ngpu))
        else:
            output = self.main(input)
        return output


class Discriminator(nn.Module):
    def __init__(self, ngpu, nc=1, ndf=64):
        super(Discriminator, self).__init__()
        self.ngpu = ngpu
        self.main = nn.Sequential(
            # input is (nc) x 64 x 64
            nn.Conv2d(nc, ndf, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf) x 32 x 32
            nn.Conv2d(ndf, ndf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 2),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*2) x 16 x 16
            nn.Conv2d(ndf * 2, ndf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 4),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*4) x 8 x 8
            nn.Conv2d(ndf * 4, 1, 4, 2, 1, bias=False),
            nn.Sigmoid()
        )

    def forward(self, input):
        if input.is_cuda and self.ngpu > 1:
            output = nn.parallel.data_parallel(self.main, input, range(self.ngpu))
        else:
            output = self.main(input)
        return output.view(-1, 1).squeeze(1)

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 20, 5, 1)
        self.conv2 = nn.Conv2d(20, 50, 5, 1)
        self.fc1 = nn.Linear(4*4*50, 500)
        self.fc2 = nn.Linear(500, 10)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.max_pool2d(x, 2, 2)
        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x, 2, 2)
        x = x.view(-1, 4*4*50)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return F.log_softmax(x, dim=1)

