
import numba
import numpy as np


class IL:
    def __init__(self, kernel):
        self.kernel = kernel

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
        return K_x @ self.beta
    
    def thres_pred(self, x, threshold):
        pred = self(x)
        pred -= threshold
        np.sign(pred, out=pred)
        return pred
    
    def topk_pred(self, x, k):
        soft_pred = self(x)
        sort_pred = np.argsort(soft_pred, axis=1)[:, -k:]
        pred = np.zeros(soft_pred.shape, dtype=np.bool_)
        self.fill_topk_pred(pred, sort_pred)
        return pred
       
    @staticmethod
    @numba.jit("(b1[:,:], i8[:,:])", nopython=True)
    def fill_topk_pred(pred, sort_pred):
        n, m = sort_pred.shape
        for i in range(n):
            for j in range(m):
                pred[i, sort_pred[i, j]] = True
                