
import numba
import numpy as np


class DF:
    def __init__(self, kernel, k):
        self.kernel = kernel
        self.k = k

    def train(self, x_train, S_train, lambd):
        """
        S_train[i,j] = 1 -> i has label j positive
        S_train[i,j] = -1 -> i has label j negative
        S_train[i,j] = 0 -> no information is given for label j of i
        """
        n_train, self.m = S_train.shape

        self.kernel.set_support(x_train)
        K = self.kernel.get_k()
        w, v = np.linalg.eigh(K)
        w_reg = w / (w + n_train * lambd)
        alpha = (w_reg * v) @ v.T

        self.y_train = self.disambiguation(alpha, S_train, self.k)
        self.beta = v / (w + n_train * lambd) @ (v.T @ self.y_train)

    def __call__(self, x):
        K_x = self.kernel(x).T
        pred = K_x @ self.beta
        idx = np.argsort(pred, axis=1)[:, -self.k:]
        pred[:] = -1
        self.fill_topk_pred(pred, idx)
        return pred    
    
    @classmethod
    def disambiguation(cls, alpha, S_train, k):
        n_train, m = S_train.shape
        const_idx = S_train != 0
        value = S_train[const_idx]
        
        y_train = np.asfortranarray(S_train, dtype=np.float)
        z = np.zeros(y_train.shape)
        z_old = np.ones(y_train.shape)
        while not (z == z_old).all():
            z_old[:] = z[:]

            np.matmul(alpha, y_train, out=z)
            idx = np.argsort(z, axis=1)[:, -k:]
            z[:] = -1
            cls.fill_topk_pred(z, idx)

            np.matmul(alpha, z, out=y_train)
            np.sign(y_train, out=y_train)
            y_train[const_idx] = value

#             y_train[S_train == 1] = np.max(y_train) + 1
#             idx = np.argsort(y_train, axis=1)[:, -k:]
#             y_train[:] = 0
#             cls.fill_topk_pred(y_train, idx)
#             y_train[const_idx] = value

        return y_train

    @staticmethod
    @numba.jit("(f8[:,:], i8[:,:])", nopython=True)
    def fill_topk_pred(pred, idx):
        n, k = idx.shape
        for i in range(n):
            for j in range(k):
                pred[i, idx[i, j]] = 1
