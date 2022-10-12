from .resnet import ResNet
from .vit import VIT
from .swin import SWIN
from .mlp_mixer import MLP_backbone


def build_model(model_name, method, use_cuda=True, use_eval=True, **kwargs):
    if 'resnet' in model_name:
        model = ResNet(method, model_name, **kwargs)
    elif 'vit' in model_name or 'deit' in model_name:
        model = VIT(method, model_name, **kwargs)
    elif 'swin' in model_name:
        model = SWIN(method, model_name, **kwargs)
    elif 'mlp' in model_name:
        model = MLP_backbone(model_name, **kwargs)
    else:
        assert 0, 'Invalid dataset name: {}'.format(model_name)
    if use_cuda:
        model = model.cuda()
    if use_eval:
        model = model.eval()
    return model
