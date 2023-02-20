import numpy as np
import numba


def scalar_product(x1, x2):
    return x1 @ x2.T


def distance_square(x1, x2):
    out = scalar_product(x1, x2)
    out *= -2
    out += np.sum(x1 ** 2, axis=1)[:, np.newaxis]
    out += np.sum(x2 ** 2, axis=1)
    out[out < 0] = 0
    return out


def distance(x1, x2):
    out = distance_square(x1, x2)
    np.sqrt(out, out=out)
    return out


def rbf_kernel(x1, x2, sigma=1):
    K = distance_square(x1, x2)
    K /= -sigma**2
    np.exp(K, out=K)
    return K


def exp_kernel(x1, x2, sigma=1):
    K = distance(x1, x2)
    K /= -sigma
    np.exp(K, out=K)
    return K


@numba.jit("f8[:, :](f8[:, :], f8[:, :], f8[:, :], f8[:])")
def _rbf_laplacian(K, S, S_in, S_out):
    p, n = K.shape
    out = np.zeros((p, p), dtype=np.float64)
    for i in range(p):
        for j in range(p):
            for k in range(n):
                tmp = S_out[k] - S[i, k] - S[j, k] + S_in[i, j]
                out[i, j] += K[i, k] * K[j, k] * tmp
    return out


@numba.jit("f8[:, :](f8[:, :], f8[:, :], f8[:, :], f8[:])")
def _exp_laplacian(K, S, S_in, S_out):
    p, n = K.shape
    out = np.zeros((p, p), dtype=np.float64)
    for i in range(p):
        for j in range(p):
            for k in range(n):
                norm = (S_out[k] - 2*S[i, k] + S_in[i, i])
                norm *= (S_out[k] - 2*S[j, k] + S_in[j, j])
                if norm < 1e-7:
                    norm = 1
                tmp = S_out[k] - S[i, k] - S[j, k] + S_in[i, j]
                out[i, j] += K[i, k] * K[j, k] * tmp / np.sqrt(norm)
    return out


def laplacian(kernel, sigma, x_repr, x, _laplacian, norm):
    K = kernel(x_repr, x, sigma=sigma)
    S_in = scalar_product(x_repr, x_repr)
    S = scalar_product(x_repr, x)
    S_out = np.sum(x ** 2, axis=1)
    L = _laplacian(K, S, S_in, S_out)
    L *= norm
    return L, K
