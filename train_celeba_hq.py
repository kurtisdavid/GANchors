import celeba_utils
import torch
import torchvision
from torchvision import transforms
import class_models
import importlib
import numpy as np
def get_accuracy(out_X, labels):
    pred = torch.argmax(out_X, dim=1).type(torch.cuda.FloatTensor)
    labels = labels.type(torch.cuda.FloatTensor)
    acc = 1 - torch.abs(pred-labels)
    acc = torch.mean(acc)
    return acc.item()

# add in augmentations whenever
data_dir = './celebA-HQ'
img_dir = 'imgs/'
labels_dir = 'Anno/'
train_transform = transforms.Compose([
                transforms.Resize((256,256)),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor()])
test_transform = transforms.Compose([
                transforms.Resize((256,256)),
                transforms.ToTensor()])

train_BS = 128
test_BS = 128
# datasets
train_set = celeba_utils.CelebA(data_dir,img_dir+'train',labels_dir + 'train.txt', transform=train_transform, mask=True)
exit(0)
print(train_set[0][0].shape)
test_set = celeba_utils.CelebA(data_dir,img_dir+'test',labels_dir + 'test.txt', transform=test_transform)

train_loader = torch.utils.data.DataLoader(train_set, batch_size = train_BS, shuffle = True, num_workers = 4)
test_loader = torch.utils.data.DataLoader(test_set, batch_size = test_BS, shuffle = False, num_workers = 4)

smiling_label = 31
device = 'cuda:0'
model = class_models.celebAModel(test_set[0][0].shape,device='cuda:0').cuda()

epochs = 5
lr = 1e-3
opt = torch.optim.Adam(lr=lr, params = list(model.parameters()))
loss_term = torch.nn.CrossEntropyLoss()
it = 0
for epoch in range(epochs):
    model.train()
    for X, label in train_loader:
        opt.zero_grad()
        X = X.to(device)
        label = label[:,smiling_label]
        loss = loss_term(model(X),label.type(torch.LongTensor).to(device))
        loss.backward()
        opt.step()
        if it % 100 == 0:
            with open('classification.txt','a') as f:
                print(it, loss.item(), file=f)
        it += 1
    with torch.no_grad():
        accs = []
        model.eval()
        for X, label in test_loader:
            X = X.to(device)
            label = label[:,smiling_label].type(torch.LongTensor).to(device)
            accs.append(get_accuracy(model(X), label))
        with open('classification.txt','a') as f:
            print("Epoch", epoch, " done | Val Accuracy:", np.mean(accs), file=f)
    torch.save(model.state_dict(), 'pretrained_celeba_smiling_' + str(epoch) + '.pt')

