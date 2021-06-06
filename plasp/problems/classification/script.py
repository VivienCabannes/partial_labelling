import os
import sys
import time

import matplotlib.pyplot as plt
import numpy as np

# mock installation of the module
sys.path.append(os.path.join('..', '..'))
from weights import RidgeRegressor, Diffusion
from dataloader import LIBSVMLoader, FoldsGenerator
from problems.classification import DF


np.random.seed(0)
plt.rcParams['text.usetex'] = True
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.serif'] = 'Times'

names = LIBSVMLoader.datasets
names = ['svmguide2', 'svmguide4', 'glass', 'iris', 'vowel', 'wine', 'vehicle']
kernel_type = 'Gaussian'  # 'Gaussian', 'Laplacian', 'Linear'
nb_folds = 8                                              # number of folds
corruptions = [0, .1, .2, .3, .4, .5, .6, .7, .8, .9, 1]  # corruptions to test
sigmas = [1e0, 1e-1, 1e-2]                      # different kernel parameters to test
lambdas = np.logspace(-6, 6, num=13)                 # different regularizations to test
mus = np.logspace(-6, 6, num=13)                  # different regularizations to test

t = time.time()
kernel_reg = Diffusion(sigma=1)

err_df, err_il, err_ac = {}, {}, {}
shape_err = (len(corruptions), nb_folds, len(sigmas), len(mus), len(lambdas))

for name in names:
    print(name)

    loader = LIBSVMLoader(name)
    x, y = loader.get_trainset()

    index_to_corrupt = np.argmax(y.sum(axis=0))

    S = np.empty((*y.shape, len(corruptions)))
    for i, corruption in enumerate(corruptions):
#         S[..., i] = loader.skewed_corruption(y, corruption, index_to_corrupt)
        S[..., i] = loader.synthetic_corruption(y, corruption)

    err_df[name]= np.empty(shape_err)
    err_il[name]= np.empty(shape_err)
    err_ac[name]= np.empty(shape_err)

    floader = FoldsGenerator(x, y, S, nb_folds=nb_folds)

    for fold in range(nb_folds):

        (x_train, S_train), (x_test, y_test) = floader()
        y_test = np.argmax(y_test, axis=1)
        kernel_reg.set_support(x_train)
        n_train, dim = x_train.shape

        for i_s, c_sigma in enumerate(sigmas):
            sigma = c_sigma * dim
            kernel_reg.update_sigma(sigma, p=100)

            for i_m, c_mu in enumerate(mus):
                mu = c_mu / np.sqrt(n_train)
                kernel_reg.update_mu(mu=mu)

                for i_l, c_lambda in enumerate(lambdas):
                    lambd = c_lambda / np.sqrt(n_train)
                    psi = lambda x: (x+lambd)**(-1)
                    kernel_reg.update_psi(psi=psi)
                    alpha = kernel_reg(x_test)
                    alpha_train = kernel_reg(x_train)
                    alpha_train += alpha_train.T

                    for i_c, corruption in enumerate(corruptions):
                        s_train = S_train[..., i_c]

                        y_train = DF.disambiguation(alpha_train, s_train.astype(np.bool_))
                        y_df = np.argmax(alpha @ y_train, axis=1)

                        y_il = np.argmax(alpha @ s_train, axis=1)

                        s_train = s_train / s_train.sum(axis=1)[:, np.newaxis]
                        y_ac = np.argmax(alpha @ s_train, axis=1)

                        err_df[name][i_c, fold, i_s, i_m, i_l] = (y_df != y_test).mean()
                        err_il[name][i_c, fold, i_s, i_m, i_l] = (y_il != y_test).mean()
                        err_ac[name][i_c, fold, i_s, i_m, i_l] = (y_ac != y_test).mean()
print(time.time() - t)

t = time.time()
kernel_reg = RidgeRegressor(kernel_type, sigma=1)

_err_df, _err_il, _err_ac = {}, {}, {}
shape_err = (len(corruptions), nb_folds, len(sigmas), len(lambdas))

for name in names:
    print(name)

    loader = LIBSVMLoader(name)
    x, y = loader.get_trainset()

    index_to_corrupt = np.argmax(y.sum(axis=0))

    S = np.empty((*y.shape, len(corruptions)))
    for i, corruption in enumerate(corruptions):
#         S[..., i] = loader.skewed_corruption(y, corruption, index_to_corrupt)
        S[..., i] = loader.synthetic_corruption(y, corruption)

    _err_df[name]= np.empty(shape_err)
    _err_il[name]= np.empty(shape_err)
    _err_ac[name]= np.empty(shape_err)

    floader = FoldsGenerator(x, y, S, nb_folds=nb_folds)

    for fold in range(nb_folds):

        (x_train, S_train), (x_test, y_test) = floader()
        y_test = np.argmax(y_test, axis=1)
        kernel_reg.set_support(x_train)
        n_train, dim = x_train.shape

        for i_s, c_sigma in enumerate(sigmas):
            sigma = c_sigma * dim
            kernel_reg.update_sigma(sigma)

            for i_l, c_lambda in enumerate(lambdas):
                lambd = c_lambda / np.sqrt(n_train)
                kernel_reg.update_lambda(lambd=lambd)
                alpha = kernel_reg(x_test)
                alpha_train = kernel_reg(x_train)
                alpha_train += alpha_train.T

                for i_c, corruption in enumerate(corruptions):
                    s_train = S_train[..., i_c]

                    y_train = DF.disambiguation(alpha_train, s_train.astype(np.bool_))
                    y_df = np.argmax(alpha @ y_train, axis=1)

                    y_il = np.argmax(alpha @ s_train, axis=1)

                    s_train = s_train / s_train.sum(axis=1)[:, np.newaxis]
                    y_ac = np.argmax(alpha @ s_train, axis=1)

                    _err_df[name][i_c, fold, i_s, i_l] = (y_df != y_test).mean()
                    _err_il[name][i_c, fold, i_s, i_l] = (y_il != y_test).mean()
                    _err_ac[name][i_c, fold, i_s, i_l] = (y_ac != y_test).mean()
print(time.time() - t)

# best = np.array([['s0l0', 's1l0', 's2l0'], ['s0l1', 's1l1', 's2l1'], ['s0l2', 's1l2', 's2l2']]).reshape(-1)
n_c = len(corruptions)
mus, stds = np.empty((len(names), 6, n_c)), np.empty((len(names), 6, n_c))
for i, name in enumerate(names):
    for j, err in zip([0, 1, 2, 3, 4, 5], [err_df, err_il, err_ac, _err_df, _err_il, _err_ac]):
        tmp = err[name].reshape((len(corruptions), nb_folds, -1))
        mu = tmp.mean(axis=1)
        ind = mu.argmin(axis=-1)
        for k in range(len(corruptions)):
            mus[i, j, k] = mu[k, ind[k]]
            stds[i, j, k] = tmp[k, :, ind[k]].std()
#         print(name, best[ind])

try:
    os.mkdir('savings')
except FileExistsError:
    pass

for i, name in enumerate(names):
    fig, ax = plt.subplots(figsize=(6, 3.5))
    c = ax.errorbar([100*i for i in corruptions], mus[i, 2], .5*stds[i, 2],
                    capsize=2, linewidth=2, capthick=2, color='C0', alpha=.5)
    b = ax.errorbar([100*i for i in corruptions], mus[i, 1], .5*stds[i, 1],
                    capsize=2, linewidth=2, capthick=2, color='C2', alpha=.5)
    a = ax.errorbar([100*i for i in corruptions], mus[i, 0], .5*stds[i, 0],
                    capsize=2, linewidth=2, capthick=2, color='C1')
    f = ax.errorbar([100*i for i in corruptions], mus[i, 5], .5*stds[i, 5],
                    capsize=2, linewidth=2, capthick=2, color='C5', alpha=.5)
    e = ax.errorbar([100*i for i in corruptions], mus[i, 4], .5*stds[i, 4],
                    capsize=2, linewidth=2, capthick=2, color='C4', alpha=.5)
    d = ax.errorbar([100*i for i in corruptions], mus[i, 3], .5*stds[i, 3],
                    capsize=2, linewidth=2, capthick=2, color='C3')
    ax.legend([a, b, c, d, e, f], ['DF', 'IL', 'AC', 'DF krr', 'IL krr', 'AC krr'],
              prop={'size':6}, ncol=3)
#     ax.legend([b, c], ['IL', 'AC'], prop={'size':9}, ncol=2)
    ax.grid()
    ax.set_title(r'``' + name[0].upper() + name[1:] + ' dataset', size=10)
    ax.set_ylabel(r'Loss', size=8)
    ax.set_xlabel(r'Corruption (in \\%)', size=8)
    ax.tick_params(axis='both', which='major', labelsize=10)
#     ax.set_yticks([])
    fig.tight_layout()
    fig.savefig(os.path.join('savings', name + '.pdf'))
