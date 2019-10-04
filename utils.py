import torch
import torchvision
from generators import GeneratorResnet
import pandas as pd


# Load a particular generator
def load_gan(args):
    # Load Generator
    if args.model_type == 'incv3':
        netG = GeneratorResnet(inception=True)
    else:
        netG = GeneratorResnet()

    print('Label: {} \t Attack: {} dependent \t Model: {} \t Distribution: {} \t Saving instance: {}'.format(args.target,
                                                                                                           args.attack_type,
                                                                                                           args.model_type,
                                                                                                           args.train_dir,
                                                                                                           args.epochs))
    if args.rl:
        netG.load_state_dict(torch.load('saved_models/netG_{}_{}_{}_{}_{}_rl.pth'.format(args.target,
                                                                                         args.attack_type,
                                                                                         args.model_type,
                                                                                         args.train_dir,
                                                                                         args.epochs)))
    else:
        netG.load_state_dict(torch.load('saved_models/netG_{}_{}_{}_{}_{}.pth'.format(args.target,
                                                                                      args.attack_type,
                                                                                      args.model_type,
                                                                                      args.train_dir,
                                                                                      args.epochs)))

    return netG


# Load ImageNet model to evaluate
def load_model(args):
    # Load Targeted Model
    if args.model_t == 'dense201':
        model_t = torchvision.models.densenet201(pretrained=True)
    elif args.model_t == 'vgg19':
        model_t = torchvision.models.vgg19(pretrained=True)
    elif args.model_t == 'vgg16':
        model_t = torchvision.models.vgg16(pretrained=True)
    elif args.model_t == 'incv3':
        model_t = torchvision.models.inception_v3(pretrained=True)
    elif args.model_t == 'res152':
        model_t = torchvision.models.resnet152(pretrained=True)
    elif args.model_t == 'res50':
        model_t = torchvision.models.resnet50(pretrained=True)
    elif args.model_t == 'sqz':
        model_t = torchvision.models.squeezenet1_1(pretrained=True)

    return model_t

############################################################
# If you have all 1000 class folders. Then using default loader is ok.
# In case you have few classes (let's 50) or collected random images in a folder
# then we need to fix the labels.
# The code below will fix the labels for you as long as you don't change "orginal imagenet ids".
# for example "ILSVRC2012_val_00019972.JPEG ... "

def fix_labels(test_set):
    val_dict = {}
    with open("val.txt") as file:
        for line in file:
            (key, val) = line.split(' ')
            val_dict[key.split('.')[0]] = int(val.strip())

    new_data_samples = []
    for i, j in enumerate(test_set.samples):
        org_label = val_dict[test_set.samples[i][0].split('/')[-1].split('.')[0]]
        new_data_samples.append((test_set.samples[i][0], org_label))

    test_set.samples = new_data_samples
    return test_set
############################################################


#############################################################
# This will fix labels for NIPS ImageNet
def fix_labels_nips(test_set, pytorch=False):

    '''
    :param pytorch: pytorch models have 1000 labels as compared to tensorflow models with 1001 labels
    '''

    filenames = [i.split('/')[-1] for i, j in test_set.samples]
    # Load provided files and get image labels and names
    image_classes = pd.read_csv("images.csv")
    image_metadata = pd.DataFrame({"ImageId": [f[:-4] for f in filenames]}).merge(image_classes, on="ImageId")
    true_classes = image_metadata["TrueLabel"].tolist()
    target_classes = image_metadata["TargetClass"].tolist()

    # Populate the dictionary: key(image path), value ([true label, target label])
    val_dict = {}
    for f, i in zip(filenames, range(len(filenames))):
        val_dict[f] = [true_classes[i], target_classes[i]]

    new_data_samples = []
    for i, j in enumerate(test_set.samples):
        org_label = val_dict[test_set.samples[i][0].split('/')[-1]][0]
        if pytorch:
            new_data_samples.append((test_set.samples[i][0], org_label-1))
        else:
            new_data_samples.append((test_set.samples[i][0], org_label))

    test_set.samples = new_data_samples

    return test_set


# Rescale image b/w (-1, +1)
def rescale(image):
    return image*2-1