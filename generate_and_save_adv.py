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
import imageio
from generators import *
from gaussian_smoothing import *
from utils import *


parser = argparse.ArgumentParser(description='Cross Data Transferability')
parser.add_argument('--train_dir', default='paintings', help='Generator Training Data: paintings, comics, ImageNet')
parser.add_argument('--test_dir', default='IN/', help='Image Directory')
parser.add_argument('--save_dir', default='adv/', help='Image Directory')
parser.add_argument('--batch_size', type=int, default=10, help='Batch size for evaluation')
parser.add_argument('--epochs', type=int, default=9, help='Saving Instance')
parser.add_argument('--eps', type=int, default=16, help='Perturbation Budget')
parser.add_argument('--model_type', type=str, default= 'incv3',  help ='Model under attack: vgg16, vgg19, incv3, res152')
parser.add_argument('--attack_type', type=str, default='img', help='Training is either img/noise dependent')
parser.add_argument('--target', type=int, default=-1, help='-1 if untargeted')
parser.add_argument('--gk', action='store_true', help='Apply Gaussian Smoothings to GAN Output')
parser.add_argument('--rl', action='store_true', help='Relativstic or Simple GAN')
args = parser.parse_args()

eps = args.eps/255


# GPU
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

# Load Generator
netG = load_gan(args)
netG.to(device)
netG.eval()


# Data
# Input dimensions: Inception takes 3x299x299
if args.model_type in ['vgg16', 'vgg19', 'res152']:
    scale_size = 256
    img_size = 224
else:
    scale_size = 300
    img_size = 299

# Setup-Data
data_transform = transforms.Compose([
    transforms.Resize(scale_size),
    transforms.CenterCrop(img_size),
    transforms.ToTensor(),
])

test_dir = args.test_dir
test_set = datasets.ImageFolder(test_dir, data_transform)
test_loader = torch.utils.data.DataLoader(test_set,batch_size=args.batch_size, shuffle=False, num_workers=4, pin_memory=True)

test_size = len(test_set)
print('Test data size:', test_size)

sourcedir = args.test_dir
targetdir = args.save_dir
all_classes = sorted(os.listdir(args.test_dir))

# Setup-Gaussian Kernel
if args.gk:
    kernel_size = 3
    pad = 2
    sigma = 1
    kernel = get_gaussian_kernel(kernel_size=kernel_size, pad=pad, sigma=sigma).cuda()


counter = 0
current_class = None
current_class_files = None
big_img = []
for i, (img, label) in enumerate(test_loader):
    img = img.to(device)

    adv = netG(img)
    if args.gk:
        adv = kernel(adv)
    adv = torch.min(torch.max(adv, img - eps), img + eps)
    adv = torch.clamp(adv, 0.0, 1.0)

    print('L-infity:{}'.format((img-adv).max()*255))


    # Courtesy of: https://github.com/rgeirhos/Stylized-ImageNet/blob/master/code/preprocess_imagenet.py
    for img_index in range(adv.size()[0]):
        source_class = all_classes[label[img_index]]
        source_classdir = os.path.join(sourcedir, source_class)
        assert os.path.exists(source_classdir)

        target_classdir = os.path.join(targetdir, source_class)
        if not os.path.exists(target_classdir):
            os.makedirs(target_classdir)

        if source_class != current_class:
            # moving on to new class:
            # start counter (=index) by 0, update list of files
            # for this new class
            counter = 0
            current_class_files = sorted(os.listdir(source_classdir))

        current_class = source_class

        target_img_path = os.path.join(target_classdir,
                                       current_class_files[counter]).replace(".JPEG", ".png")

        # if size_error == 1:
        #     big_img.append(target_img_path)

        adv_to_save = np.transpose(adv[img_index, :, :, :].detach().cpu().numpy(), (1, 2, 0)) * 255
        imageio.imwrite(target_img_path, adv_to_save.astype(np.uint8))
        # save_image(tensor=adv[img_index, :, :, :],
        #            filename=target_img_path)
        counter += 1
    #
    # del(img)
    # del(adv)
    # del(adv1)

    print('Number of Images Processed:', (i + 1) * args.batch_size)
