import os
import os.path as osp
import torch
from tqdm import tqdm

from configs import get_args
from core.data import build_data_loader
from core.models import build_model
from core.pca import PCA, flatten


if __name__ == '__main__':
    # Init
    args = get_args()
    use_cuda = True

    # Data
    data_loader = build_data_loader(args, args.data, args.imagenet_dir, shuffle=False,
                                    batch_size=args.batch_size, num_workers=args.num_workers)

    # Model
    net = build_model(args.model, args.method, layers=args.layers, no_epoch=args.epoch_num, use_cuda=use_cuda,
                      pretrained=not args.wo_pretrained, args=args, imagenet_dir=args.imagenet_dir)
    layer_labels = ['LayerE', 'LayerC']
    pca = PCA()

    # Run
    with torch.no_grad():
        # Extract features and labels
        feat_list = [[] for _ in range(len(layer_labels))]
        label_list = []
        for images, labels in tqdm(data_loader):
            feats = net.forward_last_two_layers(images.cuda())
            assert len(feats) == 2
            for j, feat in enumerate(feats):
                feat_list[j].append(flatten(feat).detach().cpu())
            label_list.append(labels)

        # Save features
        os.makedirs(args.save, exist_ok=True)
        for feat, layer in zip(feat_list, layer_labels):
            feat = torch.cat(feat, dim=0)
            saved_name = 'feat_{}_{}_{}_{}.pt'.format(args.model, args.data, layer, args.method)
            saved_path = osp.join(args.save, saved_name)
            torch.save(feat.detach().cpu(), saved_path, pickle_protocol=4)
            print('Saved to {}!'.format(saved_path))

        # Save labels
        labels = torch.cat(label_list, dim=0)
        saved_name = 'label_{}_{}.pt'.format(args.model, args.data)
        saved_path = osp.join(args.save, saved_name)
        torch.save(labels.detach().cpu(), saved_path, pickle_protocol=4)
        print('Saved to {}!'.format(saved_path))

        # PCA
        for layer in layer_labels:
            saved_path = osp.join(args.save, 'feat_{}_{}_{}_{}.pt'.format(args.model, args.data, layer, args.method))
            feat = torch.load(saved_path, map_location=torch.device('cuda'))
            eigenvalues, eigenvectors = pca.fit(flatten(feat))

            saved_name = 'eig_value_{}_{}_{}_{}.pt'.format(args.model, args.data, layer, args.method)
            saved_path = osp.join(args.save, saved_name)
            torch.save(eigenvalues.detach().cpu(), saved_path, pickle_protocol=4)
            print('Saved to {}!'.format(saved_path))

            saved_name = 'eig_vector_{}_{}_{}_{}.pt'.format(args.model, args.data, layer, args.method)
            saved_path = osp.join(args.save, saved_name)
            torch.save(eigenvectors.detach().cpu(), saved_path, pickle_protocol=4)
            print('Saved to {}!'.format(saved_path))
