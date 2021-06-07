"""
Depreciated
"""

import numba
import numpy as np
import matplotlib.pyplot as plt


def f(X):
    out = np.empty((X.shape[0], 5), dtype=np.int8)
    out[:, 0] = X[:, 0] < .33
    out[:, 1] = X[:, 0] > .67
    out[:, 2] = X[:, 1] < .33
    out[:, 3] = X[:, 1] > .67
    out[:, 4] = np.sum(out[:, :4], axis=1)
    out[:, 4] = out[:, 4] == 0
    return out


def get_problem(n_train, n_test):
    m = 5
    x_train = np.random.rand(n_train, 2)
    y_train = f(x_train)
    R = (6 * np.random.rand(n_train)).astype(np.int8)
    mu = (5 * np.random.rand(n_train, m)).astype(np.int8)
    for i in range(n_train):
        mu[i] = np.isin(np.arange(m), mu[i, :R[i]]).astype(np.int)
    # assert((mu.sum(axis=1) <= R).all())
    c_train = y_train + mu
    c_train %= 2

    num = int(np.sqrt(n_test))
    mesh = np.meshgrid(np.linspace(0, 1, num), np.linspace(0, 1, num))
    x_test = np.empty((num**2, 2), dtype=np.float64)
    x_test[:, 0] = mesh[0].flatten()
    x_test[:, 1] = mesh[1].flatten()
    y_test = f(x_test)

    y_grid = np.zeros((32, 5), dtype=np.int8)
    for i in range(32):
        tmp = np.base_repr(i, 2)
        for j in range(len(tmp)):
            y_grid[i, -j-1] = int(tmp[-j-1])

    return (x_train, c_train, R), (x_test, y_test), y_grid


def plot_problem(x, y, ax=None):
    c = np.zeros((3, len(y)), dtype=np.float64)
    c[0] += y[:, 0] * .33
    c[0] += y[:, 1] * .67
    c[1] += y[:, 2] * .33
    c[1] += y[:, 3] * .67
    c[2] += y[:, 4]
    if ax is None:
        fig, ax = plt.subplots()
        ax.scatter(x[:, 0], x[:, 1], c=c.transpose())
        ax.grid()
        ax.set_title("Fonction to learn", size=20);
        return fig, ax
    else:
        ax.scatter(x[:, 0], x[:, 1], c=c.transpose())


def get_alphas(x_train, x_test, c_sigmas, c_lambdas, regressor):
    n_train, dim = x_train.shape
    # regressor = RigdeRegressor
    computer = regressor('Gaussian', sigma=None)
    computer.set_support(x_train)

    alphas = []
    for c_sigma in c_sigmas:
        tmp = []
        sigma = c_sigma * dim
        computer.update_sigma(sigma)
        K = kernel.get_k()
        w_0, v = np.linalg.eigh(K)

        for c_lambda in c_lambdas:
            lambd = c_lambda / np.sqrt(n_train)
            computer.update_lambda(lambd)
            alpha = computer(x_test)
            tmp.append(alpha)
        alphas.append(tmp)

    return alphas

@numba.jit(nopython=True)
def Hamming_loss(A, B):
    d = np.zeros((A.shape[0], B.shape[0]), dtype=np.float64)
    for i in range(A.shape[0]):
        for j in range(B.shape[0]):
            for k in range(5):
                if A[i, k] != B[j, k]:
                    d[i, j] += 1
    return d


def plot_reconstruction(x_test, y_ac, y_il, y_test):
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(20, 7))
    plot_problem(x_test, y_ac, ax=ax1)
    plot_problem(x_test, y_il, ax=ax2)
    plot_problem(x_test, y_test, ax=ax3)
    ax1.set_title("AC Reconstruction", size=25)
    ax2.set_title("IL Reconstruction", size=25)
    ax3.set_title("Ground truth", size=25)
    ax1.grid()
    ax2.grid()
    ax3.grid()
    fig.savefig('multilabel_reconstruction.pdf')
