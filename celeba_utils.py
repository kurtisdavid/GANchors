import torch
import os
from PIL import Image
from torch.utils.data.dataset import Dataset
import numpy as np
from io import StringIO

# based off of this https://github.com/jiangqy/Customized-DataLoader/blob/master/dataset_processing.py
# example use:
#    CelebA('./celeba','img_align_celeba/train','labels/train_labels.txt')
class CelebA(Dataset):
    def __init__(self, data_path, img_filepath, filename, transform=None):
        self.img_path = os.path.join(data_path, img_filepath)
        self.transform = transform
        # reading labels from file
        label_filepath = os.path.join(data_path, filename)
        # reading img file from file
        fp = open(label_filepath, 'r')
        self.img_filename = [x.split()[0].strip() for x in fp]
        fp.close()
        s = ""
        with open(label_filepath,'r') as fp:
            for line in fp:
                s += ' '.join(line.split()[1:]) + '\n'
        s = StringIO(s)
        labels = np.loadtxt(s, dtype=np.int64)
        # desired labels... (manually set during experiments)
        print(labels.shape)
        indices = [0, 4, 5, 15, 17, 18, 24, 25, 28, 31, 39]
        self.label = labels[:,indices]

    def __getitem__(self, index):
        img = Image.open(os.path.join(self.img_path, self.img_filename[index]))
        img = img.convert('RGB')
        if self.transform is not None:
            img = self.transform(img)
        label = torch.from_numpy(self.label[index]).type(torch.FloatTensor)
        return img, label
    def __len__(self):
        return len(self.img_filename)
