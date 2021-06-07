
import os
import sys

import matplotlib.pyplot as plt
import numpy as np
from scipy.linalg import eigh

# mock module installation
sys.path.append(os.path.join('..', '..'))
from weights import Diffusion


np.random.seed(0)
plt.rcParams['text.latex.preamble'] = [r'\usepackage{newtxmath}'] 
plt.rcParams['text.usetex'] = True
plt.rcParams["font.family"] = "serif"
plt.rcParams["font.serif"] = "Times"
plt.rc('text', usetex=True)


def get_trainset(nb):
    theta = 2 * np.pi * np.random.rand(nb)
    cluster = np.random.choice(4, nb) + 1
    x = cluster * np.cos(theta)
    y = cluster * np.sin(theta)
    y1 = 2
    x1 = -2*np.sqrt(3)
    x2 = 1
    y2 = -2*np.sqrt(2)
    y3 = -1
    x3 = -np.sqrt(3)
    x4 = -1
    y4 = 0
    x_train = np.vstack((np.hstack((x1, x2, x3, x4, x)), np.hstack((y1, y2, y3, y4, y)))).T
    s_train = np.zeros(x_train.shape[0])
    s_train[0] = -1
    s_train[1] = +1
    s_train[2] = -1
    s_train[3] = +1
    return x_train, s_train


def get_x_test(num):
    x = np.linspace(-4.5, 4.5, num)
    X, Y = np.meshgrid(x, x)
    x_test = np.vstack((X.reshape(-1), Y.reshape(-1))).T
    return x_test, X, Y


def representation(x_train, X, Y, Z, hard=False):
    fig, ax = plt.subplots(1, 1, figsize=(1.8, 1.5))
    n_test = len(X)
    if hard:
        ax.pcolor(X, Y, np.sign(Z.reshape((n_test, n_test))), cmap="RdBu_r", vmin=-1, vmax=1)
    else:
        ax.pcolor(X, Y, Z.reshape((n_test, n_test)) / np.sqrt(np.mean(Z**2)),
                cmap='RdBu_r', vmin=-1.2, vmax=1.2)
    ax.scatter(x_train[4:, 0], x_train[4:, 1], color='k', s=1, zorder=2)
    ax.scatter(x_train[0, 0], x_train[0, 1], color='b', s=10, edgecolor='k', zorder=2)
    ax.scatter(x_train[1, 0], x_train[1, 1], color='r', s=10, edgecolor='k', zorder=2)
    ax.scatter(x_train[2, 0], x_train[2, 1], color='b', s=10, edgecolor='k', zorder=2)
    ax.scatter(x_train[3, 0], x_train[3, 1], color='r', s=10, edgecolor='k', zorder=2)
    ax.set_xlim(-4.5, 4.5)
    ax.set_ylim(-4.5, 4.5)
    ax.tick_params(axis='both', which='major', labelleft=False, labelbottom=False)
    ax.set_xticks([])
    ax.set_yticks([])
    return fig, ax


try:
    os.mkdir('savings')
except FileExistsError:
    pass


## Generate data

nu, nl = 2000, 4
n_train = nu + nl
n_test = 100
x_train, s_train = get_trainset(nu)
x_test, X, Y = get_x_test(n_test)


## Showing eigen values

sigma = 2e-1
mu = 1 / n_train
computer = Diffusion(sigma=sigma)
computer.set_support(x_train)
computer.update_sigma(full=True)
computer.update_mu(mu, Tikhonov=True)

n_eigen = 10
v, e = eigh(computer.A, computer.B,
            subset_by_index=[computer.A.shape[0]-n_eigen, computer.A.shape[0]-1])
S_test_T = computer.kernel.get_ST(x_test)
Se = S_test_T @ e

for i in range(1, n_eigen):
    Z = Se[...,-i]
    fig, ax = representation(x_train, X, Y, Z)
    ax.set_title(r'Eigen vector \#{} ($e_{}$)'.format(i, i), size=10)
    plt.tight_layout()
    fig.savefig(os.path.join('savings', 'eigen{}.pdf'.format(i)))


## Testing usefulness of Laplacian regularization

sigma = 5e-1
n, d = x_train.shape
mu = 1e-7

# Computation of \Sigma^{-1} \widehat{S^\star g_\rho}
computer.update_sigma(sigma=sigma, full=False, nl=n_train)
computer.update_mu(mu, Tikhonov=True)

A = computer.A + mu * computer.kernel.K
b = computer.kernel.K[..., :nl] @ s_train[:nl]
computer.c = np.linalg.solve(A, b)
Z = computer(x_test)

fig, ax = representation(x_train, X, Y, Z)
ax.set_title(r'$S(\hat \Sigma + \varepsilon)^{-1} \widehat{S^\star g_\rho}$', size=10)
fig.tight_layout()
fig.savefig(os.path.join('savings', 'S.pdf'))

# Computation of L^{-1} \widehat{S^\star g_\rho}
computer.update_sigma(full=True, nl=nl)
computer.update_mu(mu, Tikhonov=True)

A = computer.B + mu * computer.kernel.TT
b = computer.kernel.ST.transpose()[..., :nl] @ s_train[:nl]
computer.c = np.linalg.solve(A, b)
Z = computer(x_test)

fig, ax = representation(x_train, X, Y, Z)
ax.set_title(r'$S(\hat L + \varepsilon)^{-1} \widehat{S^\star g_\rho}$', size=10)
plt.tight_layout()
fig.savefig(os.path.join('savings', 'L.pdf'))


## Showing reconstruction

sigma = 2e-1
lambd, l = 1, 1
n, d = x_train.shape
mu, m, m_bis = 1/n, '1/n', 'n'

computer.update_sigma(full=False, nl=nl)
computer.update_mu(mu, Tikhonov=True)
computer.update_psi(lambd=lambd)
computer.set_phi(s_train[:nl])
Z = computer.call_with_phi(x_test)
fig, ax = representation(x_train, X, Y, Z)
ax.set_title(r'$\lambda={}$, $\mu={}$'.format(l, m), size=10)
plt.tight_layout()
fig.savefig(os.path.join('savings', 'reconstruction_{}_{}.pdf'.format(l, m_bis)))
