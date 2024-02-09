from __future__ import division

from math import radians, cos, sin, asin, sqrt

import numpy as np
import pandas as pd
import scipy.sparse as sp
import torch
from scipy.stats import nbinom, norm

rand = np.random.RandomState(0)


def load_AX(apath, xpath, device):
    A = np.load(apath)
    X = np.load(xpath)
    X = X.astype(np.float32)
    means = np.mean(X, axis=(0, 2))
    X = X - means.reshape(1, -1, 1)
    stds = np.std(X, axis=(0, 2))
    X = X / stds.reshape(1, -1, 1)

    A_wave = get_normalized_adj(A)
    A_q = torch.from_numpy(calculate_random_walk_matrix(A_wave).T.astype('float32'))
    A_h = torch.from_numpy(calculate_random_walk_matrix(A_wave.T).T.astype('float32'))

    A_wave = torch.from_numpy(A_wave)
    A_q = A_q.to(device=device)
    A_h = A_h.to(device=device)
    A_wave = A_wave.to(device=device)

    return A_wave, A_q, A_h, X, stds, means


def generate_dataset(X, num_timesteps_input, num_timesteps_output):
    """
    Takes node features for the graph and divides them into multiple samples
    along the time-axis by sliding a window of size (num_timesteps_input+
    num_timesteps_output) across it in steps of 1.
    :param X: Node features of shape (num_vertices, num_features,
    num_timesteps)
    :return:
        - Node features divided into multiple samples. Shape is
          (num_samples, num_vertices, num_features, num_timesteps_input).
        - Node targets for the samples. Shape is
          (num_samples, num_vertices, num_features, num_timesteps_output).
    """
    # Generate the beginning index and the ending index of a sample, which
    # contains (num_points_for_training + num_points_for_predicting) points
    indices = [(i, i + (num_timesteps_input + num_timesteps_output)) for i
               in range(X.shape[2] - (num_timesteps_input + num_timesteps_output) + 1)]

    # Save samples
    features, target = [], []
    for i, j in indices:
        features.append(X[:, :, i: i + num_timesteps_input].transpose((0, 2, 1)))
        target.append(X[:, :, i + num_timesteps_input: j].transpose((0, 2, 1)))

    return torch.from_numpy(np.array(features)), torch.from_numpy(np.array(target))


def generate_all(X, input_, output_, train_ratio, val_ratio):
    split_line1 = int(X.shape[2] * train_ratio)
    split_line2 = int(X.shape[2] * (val_ratio + train_ratio))
    train_original_data = X[:, :, :split_line1]
    val_original_data = X[:, :, split_line1:split_line2]
    test_original_data = X[:, :, split_line2:]

    training_input, training_target = generate_dataset(train_original_data,
                                                       num_timesteps_input=input_,
                                                       num_timesteps_output=output_)
    val_input, val_target = generate_dataset(val_original_data,
                                             num_timesteps_input=input_,
                                             num_timesteps_output=output_)
    test_input, test_target = generate_dataset(test_original_data,
                                               num_timesteps_input=input_,
                                               num_timesteps_output=output_)

    return training_input, training_target, val_input, val_target, test_input, test_target


"""
Dynamically construct the adjacent matrix
"""


def get_Laplace(A):
    """
    Returns the laplacian adjacency matrix. This is for C_GCN
    """
    if A[0, 0] == 1:
        A = A - np.diag(np.ones(A.shape[0], dtype=np.float32))  # if the diag has been added by 1s
    D = np.array(np.sum(A, axis=1)).reshape((-1,))
    D[D <= 10e-5] = 10e-5
    diag = np.reciprocal(np.sqrt(D))
    A_wave = np.multiply(np.multiply(diag.reshape((-1, 1)), A),
                         diag.reshape((1, -1)))
    return A_wave


def get_normalized_adj(A):
    A = A + np.diag(np.ones(A.shape[0], dtype=np.float32))
    D = np.array(np.sum(A, axis=1)).reshape((-1,))
    D[D <= 10e-5] = 10e-5  # Prevent infs
    diag = np.reciprocal(np.sqrt(D))
    A_wave = np.multiply(np.multiply(diag.reshape((-1, 1)), A),
                         diag.reshape((1, -1)))

    return A_wave


def calculate_random_walk_matrix(adj_mx):
    """
    Returns the random walk adjacency matrix. This is for D_GCN
    """
    adj_mx = sp.coo_matrix(adj_mx)
    d = np.array(adj_mx.sum(1))
    d_inv = np.power(d, -1).flatten()
    d_inv[np.isinf(d_inv)] = 0.
    d_mat_inv = sp.diags(d_inv)
    random_walk_mx = d_mat_inv.dot(adj_mx).tocoo()
    return random_walk_mx.toarray()


def nb_nll_loss(y, n, p, y_mask=None):
    """
    y: true values
    y_mask: whether missing mask is given
    """
    nll = torch.lgamma(n) + torch.lgamma(y + 1) - torch.lgamma(n + y) - n * torch.log(p) - y * torch.log(1 - p)
    if y_mask is not None:
        nll = nll * y_mask
    return torch.sum(nll)


def gauss_loss(y, loc, scale, y_mask=None):
    """
    The location (loc) keyword specifies the mean. The scale (scale) keyword specifies the standard deviation.
    http://jrmeyer.github.io/machinelearning/2017/08/18/mle.html
    """
    torch.pi = torch.acos(torch.zeros(1)).item() * 2  # ugly define pi value in torch format
    LL = -1 / 2 * torch.log(2 * torch.pi * torch.pow(scale, 2)) - 1 / 2 * (torch.pow(y - loc, 2) / torch.pow(scale, 2))
    return -torch.sum(LL)


def mg_nll(y, loc, scale, y_mask=None):
    y_mu = y - loc
    y_mu_t = y_mu.transpose(2, 3)  # y_mu.reshape(y_mu.shape[0],1)
    dig = torch.diag_embed(scale)  # .float()
    det = torch.det(dig)
    dig2 = dig.squeeze(2)
    inverse = torch.inverse(dig2)
    constant = 5 * torch.log(torch.tensor(torch.pi))
    c = torch.matmul(y_mu, inverse)
    d = torch.matmul(c, y_mu_t)
    e = d.squeeze(2)
    sum_ = (constant + torch.log(det) + e) / 2

    return torch.nanmean(sum_)
