# HDKD
### **HDKD: Hybrid Data-Efficient Knowledge Distillation Network for Medical Image Classification**
This is the official pytorch implementation of the [HDKD paper](https://arxiv.org/pdf/2407.07516).

<hr />

> **Abstract:** *Vision Transformers (ViTs) have achieved significant advancement in computer vision tasks
due to their powerful modeling capacity. However, their performance notably degrades when
trained with insufficient data due to lack of inherent inductive biases. Distilling knowledge
and inductive biases from a Convolutional Neural Network (CNN) teacher has emerged as
an effective strategy for enhancing the generalization of ViTs on limited datasets. Previous
approaches to Knowledge Distillation (KD) have pursued two primary paths: some focused
solely on distilling the logit distribution from CNN teacher to ViT student, neglecting the rich
semantic information present in intermediate features due to the structural differences between
them. Others integrated feature distillation along with logit distillation, yet this introduced
alignment operations that limits the amount of knowledge transferred due to mismatched architectures and increased the computational overhead. To this end, this paper presents Hybrid
Data-efficient Knowledge Distillation (HDKD) paradigm which employs a CNN teacher and
a hybrid student. The choice of hybrid student serves two main aspects. First, it leverages
the strengths of both convolutions and transformers while sharing the convolutional structure
with the teacher model. Second, this shared structure enables the direct application of feature
distillation without any information loss or additional computational overhead. Additionally, we propose an efficient light-weight convolutional block named Mobile Channel-Spatial
Attention (MBCSA), which serves as the primary convolutional block in both teacher and student models. Extensive experiments on two medical public datasets showcase the superiority
of HDKD over other state-of-the-art models and its computational efficiency*
<hr />

## Architecture Overview
<div align="center">
<img src="Images/HDKD.svg"/>
</div>


## Installation
Begin by cloning the HDKD repository and navigating to the project directory.
```bash
git clone https://github.com/omarsherif200/HDKD.git
cd HDKD
```

Create a new conda virtual environment.
```bash
conda create -n HDKD
conda activate HDKD
```

Install [Pytorch](https://pytorch.org/) and [torchvision](https://pytorch.org/vision/stable/index.html) using the following instruction.
```bash
pip install torch==2.3.1+cu121 torchvision==0.18.1+cu121 --extra-index-url https://download.pytorch.org/whl/cu121
```

Install other dependencies.
```bash
pip install -r requirements.txt
```

## Data preparation
Download your training and testing data and structure the data as follows
```bash
/path/to/dataset/
  train/
    class1/
      img1.jpeg
    class2/
      img2.jpeg
  test/
    class1/
      img3.jpeg
    class/2
      img4.jpeg
```

## Training

To train `teacher model` on your dataset 

```shell script
python main.py --batch-size 16 --model teacher_model --epochs 200 --data-path /path/to/dataset
```

To train `student model` on your dataset 

```shell script
python main.py --batch-size 16 --model student_model  --epochs 200 --data-path /path/to/dataset
```

To train `HDKD model` on your dataset 

```shell script
python main.py --batch-size 16 --model HDKD  --epochs 200 --teacher-path /path/to/teacher/checkpoints/for/distillation --data-path /path/to/dataset
```

To train with smote, you should set `use_smote = True` and you should add a JSON file called classes_distribution.json that includes each class associated with the number of augmented samples you want from smote
```bash
{"class1": N1,
 "class2": N2,
  ...........,
 "classk": Nk}
```
Note: smote usage isn't recommended in many cases such as when using large and balanced datasets so it is set by default to false (`use_smote = False`)

## Evaluation

To evaluate the model, provide the model checkpoint and the test dataset you want to use for evaluation.

```shell script
python test.py --model HDKD --checkpoint /path/to/model/checkpoint --batch-size 16 --data-path /path/to/testset
```

## 📧 Contact
if you have any question, please email `omarsherif0200@gmail.com` or `omarsherif@cis.asu.edu.eg`

## License
This project is released under the MIT license. Please see the [LICENSE](LICENSE) file for more information.

## Citation
Please consider citing our paper if you find it useful in your research
```
@article{el2024hdkd,
  title={HDKD: Hybrid Data-Efficient Knowledge Distillation Network for Medical Image Classification},
  author={EL-Assiouti, Omar S and Hamed, Ghada and Khattab, Dina and Ebied, Hala M},
  journal={arXiv preprint arXiv:2407.07516},
  year={2024}
}
```