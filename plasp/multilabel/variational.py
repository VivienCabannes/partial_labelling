
import numba
import numpy as np


class IL:
    def __init__(self, kernel, k):
        self.kernel = kernel
        self.k = k

    def train(self, x_train, S_train, lambd):
        """
        S_train[i,j] = 1 -> i has label j positive
        S_train[i,j] = -1 -> i has label j negative
        S_train[i,j] = 0 -> no information is given for label j of i
        """
        n_train = len(x_train)
        if S_train.dtype == np.bool_:
            phi = np.asfortranarray(S_train, dtype=np.float)
        else:
            phi = S_train
        self.kernel.set_support(x_train)
        K_lambda = self.kernel.get_k()
        K_lambda += lambd * n_train * np.eye(n_train)
        self.beta = np.linalg.solve(K_lambda, phi)

    def __call__(self, x):
        K_x = self.kernel(x).T
        pred = K_x @ self.beta
        idx = np.argsort(pred, axis=1)[:, -self.k:]
        pred[:] = -1
        self.fill_topk_pred(pred, idx)
        return pred    

    @staticmethod
    @numba.jit("(f8[:,:], i8[:,:])", nopython=True)
    def fill_topk_pred(pred, idx):
        n, k = idx.shape
        for i in range(n):
            for j in range(k):
                pred[i, idx[i, j]] = 1


class AC(IL):
    def __init__(self):
        super(AC, self).__init__(self)


class SP(IL):
    def __init__(self):
        super(SP, self).__init__(self)