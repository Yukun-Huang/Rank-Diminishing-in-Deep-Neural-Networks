import os
import os.path as osp
import cv2
from random import sample, choices
from torchvision import transforms
from torch.utils.data import Dataset
from torchvision.datasets.folder import is_image_file, default_loader
from timm.data.transforms import str_to_interp_mode
import math
from PIL import Image


def make_dataset(data_dir, limit=None):
    images = []
    if not osp.isdir(data_dir):
        print(data_dir)
        raise Exception('Check data dir')
    for root, dirs, files in os.walk(data_dir):
        for filename in files:
            if is_image_file(filename):
                images.append(osp.join(root, filename))
    if limit is not None:
        images = sample(images, k=limit)
    images = sorted(images)
    return images


def get_transform(train_or_test, args):
    if train_or_test == 'train':
        return transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
        ])
    elif train_or_test == 'test':
        test_size = 224
        if 'resnet' in args.model:
            return transforms.Compose([
                transforms.Resize((int(256/224*test_size), int(256/224*test_size))),
                transforms.CenterCrop(test_size),
                # transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
            ])
        else:
            interpolation = 'bicubic'
            crop_pct = 0.9
            scale_size = int(math.floor(test_size / crop_pct))
            if args.model in ['swin_t', 'swin_s', 'swin_b', 'deit_t', 'deit_s', 'deit_b']:
                return transforms.Compose([
                    transforms.Resize(scale_size, interpolation=str_to_interp_mode(interpolation)),
                    transforms.CenterCrop(test_size),
                    # transforms.Resize((224, 224)),
                    transforms.ToTensor(),
                    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                         std=[0.229, 0.224, 0.225])
                ])
            else:
                return transforms.Compose([
                    transforms.Resize(scale_size, interpolation=str_to_interp_mode(interpolation)),
                    transforms.CenterCrop(test_size),
                    # transforms.Resize((224, 224)),
                    transforms.ToTensor(),
                    transforms.Normalize(mean=[0.5, 0.5, 0.5],
                                         std=[0.5, 0.5, 0.5])
                ])
    raise NotImplementedError


class ImageNetFolder(Dataset):
    def __init__(self, root, train_or_test, args):
        assert osp.isdir(root), 'check data root: {}!'.format(root)
        classes = sorted(os.listdir(root))
        assert len(classes) == 1000, 'len(class_folders)={}'.format(len(classes))
        self.classes_to_indices = {}
        for i, class_name in enumerate(classes):
            self.classes_to_indices[class_name] = i
        self.imgs, self.labels = [], []
        for each_class in classes:
            imgs_each_class = make_dataset(osp.join(root, each_class))
            assert len(imgs_each_class) > 0, 'invalid folder!'
            self.imgs.extend(imgs_each_class)
            self.labels.extend([self.classes_to_indices[each_class]] * len(imgs_each_class))
        self.root = root
        self.transform = get_transform(train_or_test, args)
        print(self.transform)
        print('{}:  n_samples={}, n_classes={}'.format(self.__class__.__name__, len(self), len(classes)))

    def __getitem__(self, index):
        path, label = self.imgs[index], self.labels[index]
        try:
            img = default_loader(path)
        except:
            print(path)
            img = Image.fromarray(cv2.cvtColor(cv2.imread(path), cv2.COLOR_BGR2RGB))
        img = self.transform(img)
        return img, label

    def __len__(self):
        return len(self.imgs)


class SubImageNetFolder(Dataset):
    def __init__(self, root, train_or_test, args, n_classes=1000, n_instances=10):  # 1000, 5
        assert osp.isdir(root), 'check data root!'
        class_folders = os.listdir(root)
        if n_classes > len(class_folders):
            print('[WARNING] class_folders < n_classes! len(class_folders)={}'.format(len(class_folders)))
            n_classes = min(n_classes, len(class_folders))
        self.classes_to_indices = {}
        for i, class_name in enumerate(sorted(class_folders)):
            self.classes_to_indices[class_name] = i
        classes = sample(os.listdir(root), k=n_classes)
        self.imgs, self.labels = [], []
        for each_class in classes:
            imgs_each_class = make_dataset(osp.join(root, each_class))
            assert len(imgs_each_class) > 0, 'invalid folder!'
            try:
                self.imgs.extend(sample(imgs_each_class, k=n_instances))
                self.labels.extend([self.classes_to_indices[each_class]] * n_instances)
            except Exception as e:
                print(e)
                self.imgs.extend(choices(imgs_each_class, k=n_instances))
                self.labels.extend([self.classes_to_indices[each_class]] * n_instances)
        self.root = root
        self.transform = get_transform(train_or_test, args)
        print('{}:  n_samples={}, n_classes={}, n_instances={}'.format(
            self.__class__.__name__, len(self), n_classes, n_instances))

    def __getitem__(self, index):
        path, label = self.imgs[index], self.labels[index]
        img = default_loader(path)
        img = self.transform(img)
        return img, label

    def __len__(self):
        return len(self.imgs)
