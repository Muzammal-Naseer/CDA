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
import torchvision.utils as vutils

from gaussian_smoothing import *
from utils import *

parser = argparse.ArgumentParser(description='Cross Data Transferability')
parser.add_argument('--train_dir', default='paintings', help='Generator Training Data: paintings, comics, ')
parser.add_argument('--test_dir', default='../../IN/val', help='ImageNet Validation Data')
parser.add_argument('--is_nips', action='store_true', help='Evaluation on NIPS data')
parser.add_argument('--measure_adv', action='store_true', help='If not triggered then measuring only clean accuracy')
parser.add_argument('--batch_size', type=int, default=20, help='Batch Size')
parser.add_argument('--epochs', type=int, default=9, help='Which Saving Instance to Evaluate')
parser.add_argument('--eps', type=int, default=10, help='Perturbation Budget')
parser.add_argument('--model_type', type=str, default= 'vgg16',  help ='Model against GAN is trained: vgg16, vgg19, incv3, res152')
parser.add_argument('--model_t',type=str, default= 'vgg19',  help ='Model under attack : vgg16, vgg19, incv3, res152, res50, dense121, sqz' )
parser.add_argument('--target', type=int, default=-1, help='-1 if untargeted')
parser.add_argument('--attack_type', type=str, default='img', help='Training is either img/noise dependent')
parser.add_argument('--gk', action='store_true', help='Apply Gaussian Smoothings to GAN Output')
parser.add_argument('--rl', action='store_true', help='Relativstic or Simple GAN')
args = parser.parse_args()
print(args)

# Normalize (0-1)
eps = args.eps/255

# GPU
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

# Input dimensions: Inception takes 3x299x299
if args.model_type in ['vgg16', 'vgg19', 'res152']:
    scale_size = 256
    img_size = 224
else:
    scale_size = 300
    img_size = 299


if args.measure_adv:
    netG=load_gan(args)
    netG.to(device)
    netG.eval()

model_t = load_model(args)
model_t = model_t.to(device)
model_t.eval()



# Setup-Data
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

test_dir = args.test_dir
test_set = datasets.ImageFolder(test_dir, data_transform)
test_size = len(test_set)
print('Test data size:', test_size)

# Fix labels if needed
if args.is_nips:
    test_set = fix_labels_nips(test_set, pytorch=True)
else:
    test_set = fix_labels(test_set)

test_loader = torch.utils.data.DataLoader(test_set,batch_size=args.batch_size, shuffle=False, num_workers=4, pin_memory=True)


# Setup-Gaussian Kernel
if args.gk:
    kernel_size = 3
    pad = 2
    sigma = 1
    kernel = get_gaussian_kernel(kernel_size=kernel_size, pad=pad, sigma=sigma).cuda()


# Evaluation
adv_acc = 0
clean_acc = 0
fool_rate = 0
for i, (img, label) in enumerate(test_loader):
    img, label = img.to(device), label.to(device)

    clean_out = model_t(normalize(img.clone().detach()))
    clean_acc += torch.sum(clean_out.argmax(dim=-1) == label).item()



    if args.measure_adv:
        # Unrestricted Adversary
        adv = netG(img)

        # Apply smoothing
        if args.gk:
            adv = kernel(adv)

        # Projection
        adv = torch.min(torch.max(adv, img - eps), img + eps)
        adv = torch.clamp(adv, 0.0, 1.0)

        adv_out = model_t(normalize(adv.clone().detach()))
        adv_acc +=torch.sum(adv_out.argmax(dim=-1) == label).item()

        fool_rate += torch.sum(adv_out.argmax(dim=-1) != clean_out.argmax(dim=-1)).item()

        #l_inf = torch.dist(img.view(-1), adv.view(-1), float('inf'))
        if i == 0:
            vutils.save_image(vutils.make_grid(adv, normalize=True, scale_each=True), 'adv.png')
            vutils.save_image(vutils.make_grid(img, normalize=True, scale_each=True), 'org.png')

    if args.measure_adv:
        print('At Batch:{}\t l_inf:{}'.format(i, (img - adv).max() * 255))
    else:
        print('At Batch:{}'.format(i))

print('Clean:{0:.3%}\t Adversarial :{1:.3%}\t Fooling Rate:{2:.3%}'.format(clean_acc/test_size, adv_acc/test_size, fool_rate/test_size))

