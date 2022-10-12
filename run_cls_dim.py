import os.path as osp
from tqdm import tqdm
import torch
from functools import partial
from configs import get_args
from core.models import build_model
from core.utils.imagenet import calc_accuracy_each_class
from core.pca import feature_projection, classification_projection


def Effective_Dimension(ratio=0.95):
    # Original Accuracy
    logits = classification_projection(feats, net, which_layer, args.model)
    acc_ori = calc_accuracy_each_class(logits, targets, each_class=False)
    # Accuracy of Principal Components
    n_components = feats.size(1)
    for end_idx in tqdm(range(n_components)):
        feat_proj_p = feature_projection(feats, eig_vectors, start_idx=0, end_idx=end_idx)
        logits = classification_projection(feat_proj_p, net, which_layer, args.model)
        acc_pred = calc_accuracy_each_class(logits, targets, each_class=False)
        if acc_pred >= ratio * acc_ori:
            print(f'{args.model} needs {end_idx}/{n_components} components'
                  f' to reach 95% original classification accuracy!')
            break


if __name__ == '__main__':
    # Init
    args = get_args()

    # Model
    net = build_model(args.model, args.method, layers=args.layers, pretrained=not args.wo_pretrained, args=args)

    # Load Data
    which_layer = 'LayerE'
    suffix = '{}_{}_{}_{}'.format(args.model, args.data, which_layer, args.method)
    torch_load = partial(torch.load, map_location=torch.device('cuda'))
    feats = torch_load(f=osp.join(args.save, f'feat_{suffix}.pt'))
    eig_values = torch_load(f=osp.join(args.save, f'eig_value_{suffix}.pt'))
    eig_vectors = torch_load(f=osp.join(args.save, f'eig_vector_{suffix}.pt'))
    targets = torch_load(f'./resource/label_{args.data}.pt')

    # Run
    with torch.no_grad():
        Effective_Dimension(ratio=0.95)

'''
Results:
    resnet18 needs 149/512 components to reach 95% original classification accuracy!
    resnet50 needs 131/2048 components to reach 95% original classification accuracy!
    resmlp needs 196/384 components to reach 95% original classification accuracy!
    gmlpmixer_t needs 199/384 components to reach 95% original classification accuracy!
    vit_t needs 109/192 components to reach 95% original classification accuracy!
    swin_t needs 344/768 components to reach 95% original classification accuracy!
'''
