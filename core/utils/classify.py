import torch
from torch.nn import functional
import numpy as np
from collections import defaultdict


def get_sparse_param(param, n_principle, indices_sorted=None):
    if n_principle is None or n_principle == param.size(0):
        return param
    new_param = torch.zeros_like(param)
    if n_principle == 0:
        return new_param
    if indices_sorted is None:
        # Sorted Weight
        values_sorted, indices_sorted = torch.sort(torch.abs(param), dim=0, descending=True)
        indices_sorted = [elem.item() for elem in indices_sorted]
    principle_indices = indices_sorted[:n_principle]
    new_param[principle_indices] = param[principle_indices]
    return new_param


def predict_logits(feat_n, feat_p, param):
    if feat_n.shape[0] >= 5000:  # saving GPU memories
        feat_p_fake = torch.zeros_like(feat_p)
        batch_size = 25000
        if feat_n.shape[0] % batch_size != 0:
            batch_size = 50
            assert feat_n.shape[0] % batch_size == 0
        for i in range(feat_n.shape[0] // batch_size):
            feat_p_fake[i*batch_size:(i+1)*batch_size] = \
                feat_n[i*batch_size:(i+1)*batch_size, :].matmul(param)
    else:
        feat_p_fake = feat_n.matmul(param)
    return feat_p_fake


def get_errors_and_vars(predict, gt, error_type='rel'):
    errs_abs = torch.abs(predict - gt)
    errs_rel = torch.div(errs_abs, torch.abs(gt))
    err_mean_abs = torch.mean(errs_abs).item()
    err_mean_rel = torch.mean(errs_rel).item()
    var_abs = torch.var(errs_abs).item()
    var_rel = torch.var(errs_rel).item()
    if error_type == 'rel':
        return err_mean_rel, var_rel
    if error_type == 'abs':
        return err_mean_abs, var_abs


def predict_probs(feat_n, prob_p, param, n_principle=None, indices_sorted=None):
    with torch.no_grad():
        if n_principle is not None:
            assert indices_sorted
            _param = torch.zeros_like(param)
            for i in range(n_principle):
                principle_indices = indices_sorted[-n_principle:]
                _param[principle_indices] = param[principle_indices]
        else:
            _param = param
        # saving GPU memories
        if feat_n.shape[0] > 100:
            prob_fake = torch.zeros_like(prob_p)
            for i in range(feat_n.shape[0]):
                feat_fake = torch.cat((feat_n[i:i+1, :], feat_n[i:i+1, :].matmul(_param)), dim=1)
                prob_fake[i] = functional.softmax(feat_fake, dim=1)[:, -1].unsqueeze(1)
        else:
            feat_fake = torch.cat((feat_n, feat_n.matmul(_param)), dim=1)
            prob_fake = functional.softmax(feat_fake, dim=1)[:, -1].unsqueeze(1)
        return torch.mean(torch.abs(prob_fake - prob_p))


def extract_slice(feats, targets, class_i, target2index, source='pos'):
    start_idx, end_idx = target2index[class_i][0], target2index[class_i][-1]
    if source == 'all':
        feats_i = feats
        targets_i = targets
        feats_i_n = torch.cat((feats[:, :class_i], feats[:, class_i + 1:]), dim=1)
        feats_i_p = feats[:, class_i].unsqueeze(1)
    else:
        if source == 'pos':
            feats_i = feats[start_idx:end_idx+1, :]
            targets_i = targets[start_idx:end_idx + 1]
        elif source == 'neg':
            feats_i = torch.cat((feats[:start_idx, :], feats[end_idx + 1:, :]), dim=0)
            targets_i = torch.cat((targets[:start_idx], targets[end_idx + 1:]), dim=0)
        else:
            assert 0
        feats_i_n = torch.cat((feats_i[:, :class_i], feats_i[:, class_i + 1:]), dim=1)
        feats_i_p = feats_i[:, class_i].unsqueeze(1)
    return feats_i, targets_i, feats_i_n, feats_i_p


def performance_metric(feat_n, feat_p, param, target=None, mode='logit', n_principle=None, indices_sorted=None):
    sparse_param = get_sparse_param(param, n_principle, indices_sorted)
    feat_p_fake = predict_logits(feat_n, feat_p, sparse_param)
    if mode == 'logit':
        return get_errors_and_vars(feat_p_fake, feat_p, error_type='rel')
    elif mode == 'prob':
        feat = torch.cat((feat_n, feat_p), dim=1)
        prob_p = functional.softmax(feat, dim=1)[:, -1].unsqueeze(1)
        feat_fake = torch.cat((feat_n, feat_p_fake), dim=1)
        prob_p_fake = functional.softmax(feat_fake, dim=1)[:, -1].unsqueeze(1)
        return get_errors_and_vars(prob_p_fake, prob_p, error_type='rel')
    elif mode == 'acc':
        assert target is not None
        class_idx = target[0]
        feat = torch.cat((feat_n[:, :class_idx], feat_p, feat_n[:, class_idx:]), dim=1)
        feat_fake = torch.cat((feat_n[:, :class_idx], feat_p_fake, feat_n[:, class_idx:]), dim=1)
        acc = feat.max(dim=1)[1].eq(target).sum().item() / target.size(0)
        acc_fake = feat_fake.max(dim=1)[1].eq(target).sum().item() / target.size(0)
        return acc_fake, acc
