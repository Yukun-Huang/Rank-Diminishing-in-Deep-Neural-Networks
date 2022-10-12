# Rank Diminishing in Deep Neural Networks

This is the official code for NeurIPS 2022 paper "[_Rank Diminishing in Deep Neural Networks_](https://arxiv.org/abs/2206.06072)".

We perform a rigorous study on the behavior of network rank, focusing particularly on the notion of rank deficiency.

## Dependency

```bash
python=3.8
torch>=1.8.0
torchvision>=0.8.0
timm
tqdm
```

## Usage

### 1. Partial rank of the Jacobian

```bash
python rank_jacobian.py -m resnet50 --imagenet_dir /Path/to/ImageNet/ --weight_dir Path/to/Weights/
```


### 2. PCA dimension of feature spaces with perturbations

```bash
python rank_perturb.py -m resnet50 --imagenet_dir /Path/to/ImageNet/ --weight_dir Path/to/Weights/
```

### 3. Classification dimension of the final feature manifold

```bash
python extract_feature.py -m resnet50 --imagenet_dir /Path/to/ImageNet/ --weight_dir Path/to/Weights/

python run_cls_dim.py -m resnet50
```

### 4. Independence deficit

```bash
python extract_feature.py -m resnet50 --imagenet_dir /Path/to/ImageNet/ --weight_dir Path/to/Weights/

python run_deficit.py -m resnet50
```

## Citation

If you find this code useful, please kindly cite our paper.
