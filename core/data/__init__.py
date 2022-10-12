import os.path as osp
import torch.utils.data
import torchvision
from torchvision import datasets, transforms
from .imagenet import ImageNetFolder, SubImageNetFolder
from .iterator import CUDAPreFetchIter


def build_data_loader(args, dataset_name, root, batch_size=128, shuffle=False, num_workers=4, use_iterator=False):
    if dataset_name == 'imagenet-train-sub':
        data_folder = SubImageNetFolder(osp.join(root, 'train'), 'test', args)
    elif dataset_name == 'imagenet-train':
        data_folder = ImageNetFolder(osp.join(root, 'train'), 'test', args)
    elif dataset_name == 'imagenet-val':
        data_folder = ImageNetFolder(osp.join(root, 'val'), 'test', args)
    elif dataset_name == 'cifar10':
        pipline_test = transforms.Compose([
            transforms.Resize((32, 32)),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])
        data_folder = datasets.CIFAR10(root="./data", train=False, download=True, transform=pipline_test)
    elif dataset_name == 'cifar100':
        mean = (0.5070751592371323, 0.48654887331495095, 0.4409178433670343)
        std = (0.2673342858792401, 0.2564384629170883, 0.27615047132568404)
        pipline_test = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean, std)
        ])
        data_folder = torchvision.datasets.CIFAR100(root='./data', train=False, download=True, transform=pipline_test)
    else:
        assert 0, 'Invalid dataset name: {}'.format(dataset_name)
    data_loader = torch.utils.data.DataLoader(data_folder, batch_size, shuffle=shuffle, num_workers=num_workers)
    return CUDAPreFetchIter(data_loader) if use_iterator else data_loader
