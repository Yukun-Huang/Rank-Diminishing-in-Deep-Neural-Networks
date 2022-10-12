# Rank Diminishing in Deep Neural Networks

We perform a rigorous study on the behavior of network rank, focusing particularly on the notion of rank deficiency.

## Usage

### Dependency

```bash
torch>=1.8.0
torchvision>=0.8.0
timm
tqdm
```


## Partial rank of the Jacobian

```bash
python rank_jacobian.py -m resnet50 --imagenet_dir /Path/to/ImageNet/ --weight_dir Path/to/Weights/
```


## PCA dimension of feature spaces with perturbations

```bash
python rank_perturb.py -m resnet50 --imagenet_dir /Path/to/ImageNet/ --weight_dir Path/to/Weights/
```

## classification dimension of the final feature manifold

```bash
python extract_feature.py -m resnet50 --imagenet_dir /Path/to/ImageNet/ --weight_dir Path/to/Weights/
python run_cls_dim.py -m resnet50
```

## Independence deficit

```bash
python extract_feature.py -m resnet50 --imagenet_dir /Path/to/ImageNet/ --weight_dir Path/to/Weights/
python run_deficit.py -m resnet50
```
