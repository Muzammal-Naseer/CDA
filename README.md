# Cross-Domain Transferable Perturbations 
<!--[Project Page](https://muzammal-naseer.github.io/Cross-domain-perturbations/)-->

Pytorch Implementation of "Cross-Domain Transferability of Adversarial Perturbations" (NeurIPS 2019) [arXiv link](https://arxiv.org/abs/1905.11736).

### Table of Contents  
1) [Highlights](#Highlights) <a name="Highlights"/>
2) [Usage](#Usage) <a name="Usage"/>
3) [Pretrained-Generators](#Pretrained-Generators) <a name="Pretrained-Generators"/>
4) [How to Set-Up Data](#Datasets) <a name="Datasets"/>
5) [Training/Eval](#Training)  <a name="Training"/>
6) [Create-Adversarial-Dataset](#Create-Adversarial-Dataset) <a name="Create-Adversarial-Dataset"/>
7) [Citation](#Citation)  <a name="Citation"/>


## Highlights

1. The transferability of adversarial examples makes real-world attacks possible in black-box settings,
where the attacker is forbidden to access the internal parameters of the model. we propose a framework capable of launching highly transferable attacks that crafts adversarial patterns to mislead networks trained on different domains. The core of our proposed adversarial function is a generative network that is trained using a relativistic supervisory signal that enables domain-invariant perturbation.
2. We mainly focus on image classfication task but you can use our pretrained adversarial generators to test robustness of your model regardless of the task (Image classification, Segmentation, Object Detection etc.)
3. You don't need any particular setup (label etc.) to generate adversaries using our method. You can generate adversarial images of any size for any image dataset of your choice (see how to set-up data directory below).

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

## Pretrained-Generators
Download pretrained adversarial generators from [here](https://drive.google.com/open?id=1H_o90xtHYbK7M3bMMm_RtwTtCdDmaPJY) to 'saved_models' folder.

Adversarial generators are trained against following four models.
* ResNet152
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
  * You can try your own dataset as well.
  
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


## Create-Adversarial-Dataset
<p align="justify"> If you need to save adversaries for visualization or adversarial training, run the following command:

```
 python generate_and_save_adv.py --model_type incv3 --train_dir paintings --test_dir 'your_data/' --eps 255
```
You should see beautiful images (unbounded adversaries) like this:
![unbounded adversaries](/assets/adv_unbound_paintings_incv3.jpg)

## Citation
If you find our pretrained adversarial generators useful. Please consider citing our work
```
@article{naseer2019cross,
  title={Cross-Domain Transferability of Adversarial Perturbations},
  author={Naseer, Muzammal and Khan, Salman H and Khan, Harris and Khan, Fahad Shahbaz and Porikli, Fatih},
  journal={Advances in Neural Information Processing Systems},
  year={2019}
}
```
## Contact
Muzammal Naseer - muzammal.naseer@anu.edu.au 
<br/>
Suggestions and questions are welcome!


