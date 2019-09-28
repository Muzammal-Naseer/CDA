# Cross-Domain Transferable Perturbations 
[Project Page](https://muzammal-naseer.github.io/Cross-domain-perturbations/)

Pytorch Implementation of "Cross-Domian Transferability of Adversarial Perturbations" (NeurIPS 2019) [(arXiv link)](https://arxiv.org/abs/1905.11736), 

## Introduction

<p align="justify">The transferability of adversarial examples makes real-world attacks possible in black-box settings,
where the attacker is forbidden to access the internal parameters of the model. we propose a framework capable of launching highly transferable attacks that crafts adversarial patterns to mislead networks trained on different domains. The core of our proposed adversarial function is a generative network that is trained using a relativistic supervisory signal that enables domain-invariant perturbation</p>

![Learning Algo](/assets/cross_distribution.png)

## Usage
#### Installation
1. Install [pytorch](https://pytorch.org/).
2. Install python packages:
```
pip install -r requirements.txt
```
#### Clone the repository
```
git clone https:https://github.com/Muzammal-Naseer/Cross-domain-perturbations.git
cd Cross-domain-perturbations
```

### Pretrained Generators
<p align="justify">Download pretrained adversarial generators from here to 'saved_models' folder. We provided one pretrained generator over this repo.<p >


### Training/Evaluations


### Create and Save Adversarial Images

