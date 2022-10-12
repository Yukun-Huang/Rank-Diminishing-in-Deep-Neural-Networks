import torch
from torch.linalg import svd
from collections import defaultdict


def flatten(X):
    return X.view(X.size(0), -1)


def centralized(X, X_mean=None):
    assert len(X.shape) == 2
    if X_mean is None:
        X_mean = torch.mean(X, dim=0, keepdim=True)
    return X - X_mean


def calc_covariance_matrix(X, trick=True):
    n = X.shape[0]
    X = centralized(X)
    if trick:
        covariance_matrix = 1. / n * torch.matmul(X, X.T)
    else:
        covariance_matrix = 1. / n * torch.matmul(X.T, X)
    return covariance_matrix


def calc_soft_rank(singulars, tol=None, n_singulars=None):
    eps = torch.finfo(singulars.dtype).eps
    _rank = 0
    if n_singulars is None:
        n_singulars = min(singulars.size())
    if tol is None:
        tol = singulars.max() * n_singulars * eps
    else:
        tol = tol * singulars.max() * n_singulars * eps
    for singular in singulars:
        if singular >= tol:
            _rank += 1
    return _rank


class PerturbedFeatureRankEstimation:
    def __init__(self, batch_size, n_perturb, mag_perturb, tol=1.0):
        self.batch_size = batch_size
        self.n_perturb = n_perturb
        self.mag_perturb = mag_perturb
        self.ranks_per_layer = defaultdict(list)
        self.feat_dim_per_layer = defaultdict(int)
        self.tol = tol
        self.eps = torch.finfo(torch.float).eps

    def update(self, image, net, method='svd'):
        assert image.size(0) == 1
        image = image.expand(self.batch_size, *image.shape[1:])
        # Extract Features
        feats_each_layer = defaultdict(list)
        n_iteration, last_batch_size = self.n_perturb // self.batch_size, self.n_perturb % self.batch_size
        for i in range(n_iteration + 1):
            if i == n_iteration:
                if last_batch_size != 0:
                    image = image[:last_batch_size, :, :, :]
                else:
                    break
            perturb = torch.randn_like(image) * self.mag_perturb
            for j, feat in enumerate(net(image + perturb)):
                feats_each_layer[j].append(flatten(feat))
        # Rank Estimation
        for layer_key in sorted(feats_each_layer.keys()):
            # Preprocess
            X = torch.cat(feats_each_layer[layer_key], dim=0)
            # Rank Estimation
            if method == 'torch':
                soft_rank = torch.linalg.matrix_rank(calc_covariance_matrix(X), hermitian=False).item()
            elif method == 'svd':
                singulars = svd(calc_covariance_matrix(X)).S
                soft_rank = calc_soft_rank(singulars, tol=self.tol, n_singulars=min(X.size()))
            else:
                raise NotImplementedError
            self.ranks_per_layer[layer_key].append(soft_rank)
            self.feat_dim_per_layer[layer_key] = X.size(1)
        return self.get_current_ranks()

    def get_current_ranks(self):
        return [
            (self.ranks_per_layer[k][-1], self.feat_dim_per_layer[k])
            for k in sorted(self.ranks_per_layer.keys())
        ]


def classification_projection(feature, net, which_layer, model_type):
    if which_layer == 'LayerE':
        if 'resnet' in model_type:
            feature = net.base.fc(feature)
        elif 'mlp' in model_type:
            feature = net.head(feature)
        elif 'swin' in model_type or 'vit' in model_type:
            feature = net.head(feature)
    return feature


def feature_projection(X, V, start_idx=0, end_idx=None):
    n, c = X.shape
    if end_idx is None:
        end_idx = c
    X_proj = torch.zeros_like(X)
    for idx in range(start_idx, end_idx):
        eig_vec = V[:, idx].unsqueeze(-1)
        eig_vec_norm = eig_vec / torch.norm(eig_vec, p=2, keepdim=True)
        w_proj = torch.matmul(X, eig_vec_norm)
        X_proj_i = w_proj * eig_vec_norm.T
        X_proj += X_proj_i
    return X_proj


class PCA(object):
    def __init__(self, n_components=2):
        self.n_components = n_components
        self.mean = None
        self.proj_mat = None

    def fit(self, X):
        n = X.shape[0]
        X_mean = torch.mean(X, dim=0, keepdim=True)
        X = X - X_mean
        covariance_matrix = 1 / n * torch.matmul(X.T, X)
        eigenvalues, eigenvectors = torch.eig(covariance_matrix, eigenvectors=True)
        eigenvalues = torch.norm(eigenvalues, dim=1)
        idx = torch.argsort(-eigenvalues)
        eigenvalues = eigenvalues[idx]
        eigenvectors = eigenvectors[:, idx]
        proj_mat = eigenvectors[:, 0:self.n_components]
        self.mean, self.proj_mat = X_mean, proj_mat
        return eigenvalues, eigenvectors

    def transform(self, X):
        X = X - self.mean
        return X.matmul(self.proj_mat)
