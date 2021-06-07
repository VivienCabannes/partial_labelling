
import numpy as np


class DF:
    def __init__(self, weight_computer):
        self.computer = weight_computer

    def train(self, x_train, S_train,
              quadratic=False, method='FW', nb_epochs=1, **kwargs):

        self.computer.set_support(x_train)
        self.computer.train(**kwargs)
        alpha = self.computer(x_train)

        if quadratic:
            alpha = alpha.T @ alpha
            phi = self.quadratic_disambiguation(alpha, S_train, method, nb_epochs)
        else:
            phi = self.disambiguation(alpha, S_train)

        self.computer.set_phi(phi)

    def __call__(self, x):
        beta = self.computer.call_with_phi(x)
        idx = beta.argmax(axis=1)
        return idx

    @staticmethod
    def disambiguation(alpha, S_train):
        n_train, m = S_train.shape
        forbidden = np.invert(S_train)

        phi = S_train.astype(np.float)
        phi /= phi.sum(axis=1)[:, np.newaxis]

        aux_argmax = np.tile(np.arange(m), (n_train, 1))
        z = np.zeros(phi.shape)
        z_old = np.ones(phi.shape)
        while not (z == z_old).all():
            z_old[:] = z[:]

            np.matmul(alpha, phi, out=z)
            z[:] = aux_argmax == z.argmax(axis=1)[:, np.newaxis]

            np.matmul(alpha, z, out=phi)
            phi[forbidden] = -np.infty
            phi[:] = aux_argmax == phi.argmax(axis=1)[:, np.newaxis]

        return phi

    @staticmethod
    def quadratic_disambiguation(alpha, S_train, method, nb_epochs):
        n_train, m = S_train.shape
        forbidden = np.invert(S_train)

        phi = S_train.astype(np.float)
        phi /= phi.sum(axis=1)[:, np.newaxis]

#         alpha -= np.eye(n_train)
        aux_argmax = np.tile(np.arange(m), (n_train, 1))

        if method.lower() == 'bw':
            # Blockwise Frank-Wolfe
            for t in range(nb_epochs):
                i = np.random.randint(n_train)

                score = alpha[i] @ phi
                score[forbidden[i]] = -np.infty
                dir_bw = np.argmax(score)

                phi[i] *= t / (2*n_train + t)
                phi[i, dir_bw] += 2*n_train / (2*n_train + t)

        else:
            # Frank-Wolfe
            for t in range(nb_epochs):
                dir_fw = alpha @ phi
                dir_fw[forbidden] = -np.infty
                dir_fw[:] = aux_argmax == dir_fw.argmax(axis=1)[:, np.newaxis]

                dir_fw -= phi
                dir_fw *= 2 / (t + 2)
                phi += dir_fw

#         phi = aux_argmax == phi.argmax(axis=1)[:, np.newaxis]
        return phi


if __name__=="__main__":
    import os
    import sys

    sys.path.append(os.path.join('..', '..'))
    from weights import RidgeRegressor, Diffusion
    from dataloader import LIBSVMLoader, FoldsGenerator

    computer = RidgeRegressor('Gaussian', sigma=10)
    met = DF(computer)

    loader = LIBSVMLoader('dna')
    x, y = loader.get_trainset()
    # S = loader.synthetic_corruption(y, .6)
    S = loader.skewed_corruption(y, .6, 0)

    floader = FoldsGenerator(x, y, S)

    (x, S), (xt, y) = floader()
    y = y.argmax(axis=1)
    met.train(x, S, lambd=1e-4, quadratic=True, method='BW')
    # met.train(x, S, lambd=1e-4)
    y_p = met(xt)
    print('success: ', (y_p == y).mean())
