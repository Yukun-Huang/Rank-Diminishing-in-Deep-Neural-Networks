from __future__ import absolute_import
import torch
import torch.nn as nn
from torchvision.models import resnet18, resnet34, resnet50, resnet101, resnet152


class ResNet(nn.Module):

    layers = (['Layer_{}'.format(i) for i in range(19)])

    layer2feat = {
        'LayerP': 'xp',
        'Layer1': 'x1',
        'Layer2': 'x2',
        'Layer3': 'x3',
        'Layer4': 'x4',
        'LayerE': 'xf',
        'LayerC': 'y',
    }

    def __init__(self, method, model_name, layers=None, args=None, imagenet_dir=None, **kwargs):
        super(ResNet, self).__init__()
        factory = {
            'resnet18': resnet18,
            'resnet34': resnet34,
            'resnet50': resnet50,
            'resnet101': resnet101,
            'resnet152': resnet152,
        }
        if kwargs['pretrained'] and method == 'vanilla':
            self.base = factory[model_name](pretrained=True)
            print(f'Load {model_name} weights from torch official repo.')
        else:
            self.base = factory[model_name](pretrained=False)
            if kwargs['pretrained']:
                if 'cifar100' in method:
                    self.base = factory[model_name](pretrained=False)
                    self.base.fc = nn.Linear(512, 100, bias=True)
                    para_name = 'resnet18-200-regular.pth'
                    model_para = torch.load(para_name)
                elif method == 'mealv2':
                    self.base = factory[model_name](pretrained=False)
                    para_name = 'MEALV2_ResNet{}_224.pth'.format(model_name.replace('resnet', ''))
                    model_para = torch.load(para_name)
                    model_para = {k.replace('module.', ''): v for k, v in model_para.items()}
                else:
                    assert 0, 'Invalid method name: {}'.format(method)
                self.base.load_state_dict(model_para)
                print("loading weight from {}".format(para_name))
        if layers is not None:
            if isinstance(layers, str):
                layers = (layers,)
            assert isinstance(layers, tuple) or isinstance(layers, list)
            self.layers = tuple(layers)

    def forward(self, x, preprocess=None):
        if preprocess is not None:
            x = preprocess(x)

        feats = []
        x = self.base.conv1(x)
        x = self.base.bn1(x)
        x = self.base.relu(x)
        x = self.base.maxpool(x)

        feats.append(x)
        for i, layer1 in enumerate(self.base.layer1):
            x = layer1(x)
            feats.append(x)

        for i, layer2 in enumerate(self.base.layer2):
            x = layer2(x)
            feats.append(x)

        for i, layer3 in enumerate(self.base.layer3):
            x = layer3(x)
            feats.append(x)

        for i, layer4 in enumerate(self.base.layer4):
            x = layer4(x)
            feats.append(x)

        xf = torch.flatten(self.base.avgpool(x), 1)
        feats.append(xf)

        y = self.base.fc(xf)
        feats.append(y)

        return tuple(feats)

    def forward_last_two_layers(self, x, preprocess=None):
        if preprocess is not None:
            x = preprocess(x)

        x = self.base.conv1(x)
        x = self.base.bn1(x)
        x = self.base.relu(x)
        xp = self.base.maxpool(x)

        x1 = self.base.layer1(xp)
        x2 = self.base.layer2(x1)
        x3 = self.base.layer3(x2)
        x4 = self.base.layer4(x3)

        feats = []

        x = self.base.avgpool(x4)
        xf = torch.flatten(x, 1)
        y = self.base.fc(xf)

        feats.append(xf)
        feats.append(y)
        return tuple(feats)

    def get_layer_labels(self):
        return self.layers
