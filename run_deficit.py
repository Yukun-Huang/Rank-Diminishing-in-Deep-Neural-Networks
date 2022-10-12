import os
import os.path as osp
import torch
import torch.nn as nn
from torch.nn import functional
from tqdm import tqdm
from collections import defaultdict
from functools import partial

from configs import get_args
from core.models import build_model
from core.utils.classify import extract_slice


def calc_loss(feat_n, feat_p, param, w_reg=20.):
    loss = functional.mse_loss(feat_n.matmul(param), feat_p.detach())
    reg = torch.norm(param, p=1)
    return loss + w_reg * reg


def Train_Params(saved_folder='logs_deficit', n_iteration=5000):
    saved_folder = osp.join(saved_folder, args.model)
    os.makedirs(saved_folder, exist_ok=True)
    for class_i in tqdm(sorted(target2indices.keys())):
        feats_i, targets_i, feats_i_n, feats_i_p = extract_slice(feats, targets, class_i, target2indices)
        param_i = nn.Parameter(torch.zeros(feats_i_n.shape[1], 1).to(feats_i.device))
        for j in range(n_iteration):
            loss_i = calc_loss(feats_i_n, feats_i_p, param_i)
            loss_i.backward()
            param_i.data.sub_(args.lr * param_i.grad)
            param_i.grad.zero_()
        torch.save(param_i, osp.join(saved_folder, f'param_{class_i:04d}.pt'))


if __name__ == '__main__':
    # Init
    args = get_args()
    torch_load = partial(torch.load, map_location=torch.device('cuda'))

    # Load Data
    targets = torch_load(f'./resource/label_{args.data}.pt')
    target2indices = defaultdict(list)
    for sample_idx, target in enumerate(targets):
        target2indices[target.item()].append(sample_idx)
    suffix = '{}_{}_{}_{}'.format(args.model, args.data, 'LayerC', args.method)
    feats = torch_load(f=osp.join(args.save, f'feat_{suffix}.pt'))
    eig_values = torch_load(f=osp.join(args.save, f'eig_value_{suffix}.pt'))
    eig_vectors = torch_load(f=osp.join(args.save, f'eig_vector_{suffix}.pt'))

    # Model
    net = build_model(args.model, args.method, layers=args.layers, no_epoch=args.epoch_num, use_cuda=True,
                      pretrained=not args.wo_pretrained, args=args)

    # Optimization
    Train_Params()
