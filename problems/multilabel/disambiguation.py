
import numba
import numpy as np


class DF:
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
        alpha = self.computer(x_train)
        phi = self.disambiguation(alpha, S_train, self.k)
        self.computer.set_phi(phi)

    def __call__(self, x):
        pred = self.computer.call_with_phi(x)
        idx = np.argsort(pred, axis=1)[:, -self.k:]
        pred[:] = -1
        self.fill_topk_pred(pred, idx)
        return pred

    @classmethod
    def disambiguation(cls, alpha, S_train, k):
        n_train, m = S_train.shape
        const_idx = S_train != 0
        value = S_train[const_idx]

        phi = np.asfortranarray(S_train, dtype=np.float)
        z = np.zeros(phi.shape)
        z_old = np.ones(phi.shape)
        while not (z == z_old).all():
            z_old[:] = z[:]

            np.matmul(alpha, phi, out=z)
            idx = np.argsort(z, axis=1)[:, -k:]
            z[:] = -1
            cls.fill_topk_pred(z, idx)

            np.matmul(alpha, z, out=phi)
            np.sign(phi, out=phi)
            phi[const_idx] = value

#             phi[S_train == 1] = np.max(phi) + 1
#             idx = np.argsort(phi, axis=1)[:, -k:]
#             phi[:] = 0
#             cls.fill_topk_pred(phi, idx)
#             phi[const_idx] = value

        return phi

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
    met = DF(computer, k=1)

    loader = MULANLoader('scene')
    x, y = loader.get_trainset()
    S = loader.synthetic_corruption(y, .6)

    floader = FoldsGenerator(x, y, S)

    (x, S), (xt, y) = floader()
    met.train(x, S, lambd=1e-3)
    y_p = met(xt)
    print('KRR: ', (y_p == y).mean())

    computer = Diffusion(sigma=10)
    met = DF(computer, k=1)
    met.train(x, S, lambd=1e-2, mu=1e-4)
    y_p = met(xt)
    print('LAP: ', (y_p == y).mean())

    print('Only negative: ', (y==-1).mean())
