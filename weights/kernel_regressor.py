
import numpy as np


class KernelRegressor:
    """
    Useful to try several regularization parameter
    Based on linked between Tikhonov regularization and GSVD
    """
    def __init__(self, kernel):
        self.kernel = kernel

    def set_support(self, x_train):
        """Need to reupdate sigma and lambda after call"""
        self.x_train = x_train
        self.n_train = len(x_train)

    def update_sigma(self, sigma):
        """Need to reupdate lambda after call"""
        self.kernel.__init__(self.kernel.kernel, sigma=sigma)
        self.kernel.set_support(self.x_train)
        K = self.kernel.get_k()
        self.w_0, self.v = np.linalg.eigh(K)

    def update_lambda(self, lambd):
        w = self.w_0 + self.n_train * lambd
        w **= -1
        self.K_inv = (self.v * w) @ self.v.T

    def __call__(self, x_test):
        K_x = self.kernel(x_test)
        return K_x.T @ self.K_inv
