
import argparse
import os
import numpy as np
import matplotlib.pyplot as plt

import torchvision
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import torchvision.datasets as datasets
import torchvision.transforms as transforms

from utils import *

parser = argparse.ArgumentParser(description='Cross Data Transferability')
parser.add_argument('--train_dir', default='paintings', help='paintings, comics, imagenet')
parser.add_argument('--batch_size', type=int, default=15, help='Number of trainig samples/batch')
parser.add_argument('--epochs', type=int, default=10, help='Number of training epochs')
parser.add_argument('--lr', type=float, default=0.0002, help='Initial learning rate for adam')
parser.add_argument('--eps', type=int, default=10, help='Perturbation Budget')
parser.add_argument('--model_type', type=str, default='incv3',
                    help='Model against GAN is trained: vgg16, vgg19, incv3, res152')
parser.add_argument('--attack_type', type=str, default='img', help='Training is either img/noise dependent')
parser.add_argument('--target', type=int, default=-1, help='-1 if untargeted')
args = parser.parse_args()
print(args)

# Normalize (0-1)
eps = args.eps/255

# GPU
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

####################
# Model
####################
if args.model_type == 'vgg16':
    model = torchvision.models.vgg16(pretrained=True)
elif args.model_type == 'vgg19':
    model = torchvision.models.vgg19(pretrained=True)
elif args.model_type == 'incv3':
    model = torchvision.models.inception_v3(pretrained=True)
elif args.model_type == 'res152':
    model = torchvision.models.resnet152(pretrained=True)
model = model.to(device)
model.eval()

# Input dimensions
if args.model_type in ['vgg16', 'vgg19', 'res152']:
    scale_size = 256
    img_size = 224
else:
    scale_size = 300
    img_size = 299


# Generator
if args.model_type == 'incv3':
    netG = GeneratorResnet(inception=True)
else:
     netG = GeneratorResnet()
netG.to(device)

# Optimizer
optimG = optim.Adam(netG.parameters(), lr=args.lr, betas=(0.5, 0.999))

# Data
data_transform = transforms.Compose([
    transforms.Resize(scale_size),
    transforms.CenterCrop(img_size),
    transforms.ToTensor(),
])

mean = [0.485, 0.456, 0.406]
std = [0.229, 0.224, 0.225]
def normalize(t):
    t[:, 0, :, :] = (t[:, 0, :, :] - mean[0])/std[0]
    t[:, 1, :, :] = (t[:, 1, :, :] - mean[1])/std[1]
    t[:, 2, :, :] = (t[:, 2, :, :] - mean[2])/std[2]

    return t

train_dir = args.train_dir
train_set = datasets.ImageFolder(train_dir, data_transform)
train_loader = torch.utils.data.DataLoader(train_set, batch_size=args.batch_size, shuffle=True, num_workers=4, pin_memory=True)
train_size = len(train_set)
print('Training data size:', train_size)


# Loss
criterion = nn.CrossEntropyLoss()


####################
# Set-up noise if required
####################
if args.attack_type == 'noise':
    noise = np.random.uniform(0, 1, img_size * img_size * 3)
    # Save noise
    np.save('saved_models/noise_{}_{}_{}_rl'.format(args.target, args.model_type, args.train_dir), noise)
    im_noise = np.reshape(noise, (3, img_size, img_size))
    im_noise = im_noise[np.newaxis, :, :, :]
    im_noise_tr = np.tile(im_noise, (args.batch_size, 1, 1, 1))
    noise_tensor_tr = torch.from_numpy(im_noise_tr).type(torch.FloatTensor).to(device)


# Training
print('Label: {} \t Attack: {} dependent \t Model: {} \t Distribution: {} \t Saving instances: {}'.format(args.target, args.attack_type, args.model_type, args.train_dir, args.epochs))
for epoch in range(args.epochs):
    running_loss = 0
    for i, (img, _) in enumerate(train_loader):
        img = img.to(device)

        # whatever the model think about the input
        label = model(normalize(img.clone().detach())).argmax(dim=-1).detach()
        
        if args.target == -1:
            targte_label = torch.LongTensor(img.size(0))
            targte_label.fill_(args.target)
            targte_label = targte_label.to(device)

        netG.train()
        optimG.zero_grad()

        if args.attack_type == 'noise':
            adv = netG(noise_tensor_tr)
        else:
            adv = netG(img)

        # Projection
        adv = torch.min(torch.max(adv, img - eps), img + eps)
        adv = torch.clamp(adv, 0.0, 1.0)

        if args.target == -1:
            # Gradient accent (Untargetted Attack)
            adv_out = model(normalize(adv))
            img_out = model(normalize(img))

            loss = -criterion(adv_out-img_out, label)

        else:
            # Gradient decent (Targetted Attack)
            # loss = criterion(model(normalize(adv)), targte_label)
            loss = criterion(model(normalize(adv)), targte_label) + criterion(model(normalize(img)), label)
        loss.backward()
        optimG.step()

        if i % 10 == 9:
            print('Epoch: {0} \t Batch: {1} \t loss: {2:.5f}'.format(epoch, i, running_loss / 100))
            running_loss = 0
        running_loss += abs(loss.item())

    torch.save(netG.state_dict(), 'saved_models/netG_{}_{}_{}_{}_{}_rl.pth'.format(args.target, args.attack_type, args.model_type, args.train_dir, epoch))

    # Save noise
    if args.attack_type == 'noise':
        # Save transformed noise
        t_noise = netG(torch.from_numpy(im_noise).type(torch.FloatTensor).to(device))
        t_noise_np = np.transpose(t_noise[0].detach().cpu().numpy(), (1,2,0))
        f = plt.figure()
        plt.imshow(t_noise_np, interpolation='spline16')
        plt.xticks([])
        plt.yticks([])
        #plt.show()
        f.savefig('saved_models/noise_transformed_{}_{}_{}_{}_rl'.format(args.target, args.model_type, args.train_dir, epoch) + ".pdf", bbox_inches='tight')
        np.save('saved_models/noise_transformed_{}_{}_{}_{}_rl'.format(args.target, args.model_type, args.train_dir, epoch), t_noise_np)
