# Cross-Domain Transferable Perturbations 
[Project Page](https://muzammal-naseer.github.io/Cross-domain-perturbations/)

Pytorch Implementation of "Cross-Domian Transferability of Adversarial Perturbations" (NeurIPS 2019) [(arXiv link)](https://arxiv.org/abs/1905.11736), 

## Introduction

<p align="justify">The transferability of adversarial examples makes real-world attacks possible in black-box settings,
where the attacker is forbidden to access the internal parameters of the model. we propose a framework capable of launching highly transferable attacks that crafts adversarial patterns to mislead networks trained on different domains. The core of our proposed adversarial function is a generative network that is trained using a relativistic supervisory signal that enables domain-invariant perturbation</p>

![Learning Algo](/assets/cross_distribution.png)

## Usage
#### Dependencies
1. Install [pytorch](https://pytorch.org/).
2. Install python packages using following command:
```
pip install -r requirements.txt
```
#### Clone the repository.
```
git clone https:https://github.com/Muzammal-Naseer/Cross-domain-perturbations.git
cd Cross-domain-perturbations
```

## Pretrained Generators
<p align="justify">Download pretrained adversarial generators from here to 'saved_models' folder. We provided one pretrained generator over this repo.<p >
  
## Datasets
* For training, we used the following datasets:
  * [ImageNet](http://www.image-net.org/) Training Set.
  * [Paintings](https://www.kaggle.com/c/painter-by-numbers)
  * [Comics](https://www.kaggle.com/cenkbircanoglu/comic-books-classification)
  * [Chexnet](https://stanfordmlgroup.github.io/projects/chexnet/)
  * You can try your own data set as well
  
* For evaluations, we used the following dataset:
  * [ImageNet]((http://www.image-net.org/)) Validation Set (50k images).
  * A [subset](https://github.com/LiYingwei/Regional-Homogeneity/tree/master/data) of ImageNet validation set (5k images).
  * NeurIPS dataset (1k images).
  
* Data directory structure should look like this:
    '''bash
    |Root
        |ClassA
                img1
                img2
         .
         .
         .
     '''


## Training/Evaluations


## Create and Save Adversarial Images

