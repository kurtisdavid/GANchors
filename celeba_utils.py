import torch
import os
from PIL import Image
from torch.utils.data.dataset import Dataset
import numpy as np
from io import StringIO
import glob
import random
import numpy as np
from torchvision import transforms
# based off of this https://github.com/jiangqy/Customized-DataLoader/blob/master/dataset_processing.py
# example use:
#    CelebA('./celeba','img_align_celeba/train','labels/train_labels.txt')
class CelebA(Dataset):
    def __init__(self, data_path, img_filepath, filename, transform=None,
                 mask=False, maskpath='CelebAMask-HQ/CelebAMask-HQ-mask-anno', mask_transform=transforms.ToTensor()):
        self.img_path = os.path.join(data_path, img_filepath)
        self.transform = transform
        # reading labels from file
        label_filepath = os.path.join(data_path, filename)
        # reading img file from file
        fp = open(label_filepath, 'r')
        self.img_filename = [x.split()[0].strip() for x in fp]
        fp.close()
        # setup cache
        self.img_masks = {}
        self.mask = mask
        self.maskpath = os.path.join(data_path,maskpath)
        self.mask_transform = mask_transform
        s = ""
        with open(label_filepath,'r') as fp:
            for line in fp:
                s += ' '.join(line.split()[1:]) + '\n'
        s = StringIO(s)
        labels = np.loadtxt(s, dtype=np.int64)
        print(labels.shape)
        self.label = labels

    def __getitem__(self, index):
        img = Image.open(os.path.join(self.img_path, self.img_filename[index]))
        img = img.convert('RGB')
        if self.transform is not None:
            img = self.transform(img)
        if self.mask:
            if index not in self.img_masks:
                img_name = self.img_filename[index].split('.jpg')[0]
                folder = str(int(img_name)//2000)
                mask_files = glob.glob(os.path.join(self.maskpath,folder,img_name) + '*')
                self.img_masks[index] = np.array(mask_files)
            length = self.img_masks[index].shape[0]
            r = np.random.rand(length)
            chosen = self.img_masks[index][r > 0.7].tolist()
            final_mask = None
            for filename in chosen:
                mask = Image.open(os.path.join(filename))
                if self.mask_transform is not None:
                    mask = self.transform(mask)
                    if final_mask is None:
                        final_mask = mask
                    else:
                        final_mask += mask
            if final_mask is not None:
                final_mask = torch.clamp(final_mask,0,1)
        label = torch.from_numpy(self.label[index]).type(torch.FloatTensor)
        if self.mask:
            return img, label, final_mask
        else:
            return img, label
    def __len__(self):
        return len(self.img_filename)
