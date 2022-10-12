import os
import numpy as np
import torch
from configs import get_args
from core.data import build_data_loader
from core.models import build_model
from core.pca import PerturbedFeatureRankEstimation
from core.utils.logger import Logger


def main_perturb(n_samples=100):

    for i, (images, _) in enumerate(data_loader, start=1):

        rank_of_layers = estimator.update(images.cuda(), net)

        for j, (layer, (soft_rank, rank_upper)) in enumerate(zip(layers, rank_of_layers)):
            ranks[j].append(soft_rank)

        log_print.write('No {}: '.format(i)+','.join(['{:.2f}'.format(np.mean(rank)) for rank in ranks])+'\n')

        if i == n_samples:
            break

    log_print.write('n_samples is {}: '.format(len(ranks[0])) + ','.join(
        ['{:.2f}'.format(np.mean(rank)) for rank in ranks]) + '\n')


if __name__ == '__main__':
    # Init
    args = get_args()
    use_cuda = True

    # Data
    data_loader = build_data_loader(args, args.data, args.imagenet_dir, shuffle=True,
                                    batch_size=1, num_workers=args.num_workers)

    # Model
    net = build_model(args.model, args.method, no_epoch=args.epoch_num, use_cuda=use_cuda,
                      pretrained=not args.wo_pretrained, args=args)
    with torch.no_grad():
        test_inputs = torch.rand(1, 3, 224, 224)
        num_layers = len(net(test_inputs.cuda() if use_cuda else test_inputs))
        layers = ['Layer{}'.format(i) for i in range(num_layers)]
    ranks = [[] for i in range(num_layers)]

    # Logging
    log_path = 'logs_feat/{}_{}/{}_feat_rank.txt'.format(str(args.data), args.model, args.model)
    if args.wo_pretrained:
        log_path = log_path.replace('.txt', '_wo_pretrained.txt')
    log_print = Logger(log_path)
    log_print.write('logger successful\n')
    log_print.write(str(layers) + '\n')

    # Run
    estimator = PerturbedFeatureRankEstimation(
        batch_size=args.batch_size,
        n_perturb=5000,
        mag_perturb=1e-3,
        tol=args.tol,
    )
    with torch.no_grad():
        main_perturb()
