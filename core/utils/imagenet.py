import os
import os.path as osp
import random


def load_class_idx_to_label(txt_path='./resource/imagenet1000_clsidx_to_labels.txt'):
    txt_str = ''
    with open(txt_path, 'r') as fio:
        for line in fio.readlines():
            txt_str += line.strip('\n\r')
    return eval(txt_str)


def load_folder_idx_to_label(txt_path='./resource/imagenet1000_folder_to_labels.txt'):
    _dict = {}
    with open(txt_path, 'r') as fio:
        for line in fio.readlines():
            v, k = line.strip('\n\r').split(': ')
            _dict[k] = v
    return _dict


def calc_accuracy_each_class(logits, targets, each_class=True):
    corrects = logits.max(dim=1)[1].eq(targets)
    total_acc = corrects.sum().item() / targets.size(0)
    if not each_class:
        return total_acc
    acc_dict = {}
    cnt_dict = {}
    for correct, target in zip(corrects, targets):
        target = target.item()
        if target not in acc_dict.keys():
            acc_dict[target] = correct.item()
            cnt_dict[target] = 1
        else:
            acc_dict[target] += correct.item()
            cnt_dict[target] += 1
    for k in acc_dict.keys():
        acc_dict[k] /= cnt_dict[k]
    return total_acc, [v for k, v in sorted(acc_dict.items())]
