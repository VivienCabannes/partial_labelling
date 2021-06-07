
import numba
import numpy as np


class IL:
    def __init__(self, computer, k):
        self.computer = computer
        self.k = k

    def train(self, x_train, S_train, **kwargs):
        """
        S_train[i,j] = 1 -> i has label j positive
        S_train[i,j] = -1 -> i has label j negative
        S_train[i,j] = 0 -> no information is given for label j of i
        """
        self.computer.set_support(x_train)
        self.computer.train(**kwargs)
        if S_train.dtype == np.bool_:
            phi = np.asfortranarray(S_train, dtype=np.float)
        else:
            phi = S_train
        self.computer.set_phi(phi)

    def __call__(self, x):
        pred = self.computer.call_with_phi(x)
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


if __name__=="__main__":
    import os
    import sys

    sys.path.append(os.path.join('..', '..'))
    from weights import RidgeRegressor, Diffusion
    from dataloader import MULANLoader, FoldsGenerator

    computer = RidgeRegressor('Gaussian', sigma=100)
    met = IL(computer, k=1)

    loader = MULANLoader('scene')
    x, y = loader.get_trainset()
    S = loader.synthetic_corruption(y, .6)

    floader = FoldsGenerator(x, y, S)

    (x, S), (xt, y) = floader()
    met.train(x, S, lambd=1e-3)
    y_p = met(xt)
    print('KRR: ', (y_p == y).mean())

    computer = Diffusion(sigma=10)
    met = IL(computer, k=1)
    met.train(x, S, lambd=1e-2, mu=1e-4)
    y_p = met(xt)
    print('LAP: ', (y_p == y).mean())

    print('Only negative: ', (y==-1).mean())
