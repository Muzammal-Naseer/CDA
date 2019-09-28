# Cross-Domain Transferable Perturbations 
<!--[Project Page](https://muzammal-naseer.github.io/Cross-domain-perturbations/)

Pytorch Implementation of "Cross-Domian Transferability of Adversarial Perturbations" (NeurIPS 2019) [(arXiv link)](https://arxiv.org/abs/1905.11736), 

## Highlights

1. The transferability of adversarial examples makes real-world attacks possible in black-box settings,
where the attacker is forbidden to access the internal parameters of the model. we propose a framework capable of launching highly transferable attacks that crafts adversarial patterns to mislead networks trained on different domains. The core of our proposed adversarial function is a generative network that is trained using a relativistic supervisory signal that enables domain-invariant perturbation.
2. **We mainly focus on image classfication task but you can use our pretrained adversarial generators to test robustness of your model regardless of the task (Image classification, Segmentation, Object Detection etc.)**

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

Adversarial generators are trained against following four models.
* ResNet15
* Inceptionv3
* VGG19
* VGG16
These models are trained on ImageNet and available in Pytorch.
  
## Datasets
* Training data:
  * [ImageNet](http://www.image-net.org/) Training Set.
  * [Paintings](https://www.kaggle.com/c/painter-by-numbers)
  * [Comics](https://www.kaggle.com/cenkbircanoglu/comic-books-classification)
  * [Chexnet](https://stanfordmlgroup.github.io/projects/chexnet/)
  * You can try your own data set as well.
  
* Evaluations data:
  * [ImageNet](http://www.image-net.org/) Validation Set (50k images).
  * [Subset](https://github.com/LiYingwei/Regional-Homogeneity/tree/master/data) of ImageNet validation set (5k images).
  * [NeurIPS dataset](https://www.kaggle.com/c/nips-2017-non-targeted-adversarial-attack) (1k images).
  
* Directory structure should look like this:
 ```
    |Root
        |ClassA
                img1
                img2
                ...
        |ClassB
                img1
                img2
                ...
```
## Training
<p align="justify"> Run the following command

```
  python train.py --model_type res152 --train_dir paintings --eps 10 --rl
  
```
This will start trainig a generator trained on Paintings (--train_dir) against ResNet152 (--model_type) under perturbation budget 10 (--eps) with relativistic supervisory signal.<p>
## Evaluations
<p align="justify"> Run the following command

```
  python eval.py --model_type res152 --train_dir imagenet --test_dir ../IN/val --epochs 0 --model_t vgg19 --eps 10 --measure_adv --rl
  
```
This will load a generator trained on ImageNet (--train_dir) against ResNet152 (--model_type) and evaluate clean and adversarial accuracy of VGG19 (--model_t) under perturbation budget 10 (--eps). <p>


## Create and Save Adversarial Images
<p align="justify"> If you need to save adversaries for visualization or adversarial training, run the following command:

```
 python generate_and_save_adv.py --model_type incv3 --train_dir paintings --test_dir 'your_data/' --eps 255
  
```
You should see beautiful images like this:
<p align="center">
<img src="https://github.com/Muzammal-Naseer/Cross-domain-perturbations/blob/gh-pages/resources/images/adv_unbound_paintings_incv3.jpg"/>
</p>

## Citation
```
@article{naseer2019cross,
  title={Cross-Domain Transferability of Adversarial Perturbations},
  author={Naseer, Muzammal and Khan, Salman H and Khan, Harris and Khan, Fahad Shahbaz and Porikli, Fatih},
  journal={arXiv preprint arXiv:1905.11736},
  year={2019}
}
```



