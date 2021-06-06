
import numpy as np


class IL:
    def __init__(self, weight_computer, projection):
        self.computer = weight_computer
        self.projection = projection

    def train(self, x_train, S_train, **kwargs):
        self.computer.set_support(x_train)
        self.computer.train(**kwargs)
        self.S_train = S_train

    def __call__(self, x, tol=1e-6, max_it=1e4):
        alpha = self.computer(x)
        return self.disambiguation(alpha, self.S_train, tol, max_it, self.projection)

    @staticmethod
    def disambiguation(alpha, S_train, tol, max_it, projection):
        y_init = S_train[0]
        z = np.empty((len(alpha), y_init.shape[1]), dtype=y_init.dtype)

        for i in range(len(alpha)):
            y = y_init.copy()
            y_old = np.zeros(y.shape)
            it = 0
            while np.max(np.abs(y - y_old)) > tol and it < max_it:
                it += 1
                y_old[:] = y[:]
                z[i] = alpha[i] @ y
                y = projection(z[i], alpha[i], S_train)
            z[i] = alpha[i] @ y
            if not i % 10:
                print(i, end=',')
        return z

    @staticmethod
    def projection_ir(z, alpha, Ss):
        cs, rs = Ss
        ys = np.full(cs.shape, z)
        diff = (cs - z).squeeze()
        outside = np.abs(diff) > rs
        ys[outside] = cs[outside] - (rs[outside] * np.sign(diff)[outside])[:, np.newaxis]
        repulse = alpha < 0
        ys[repulse] = cs[repulse] + (rs[repulse] * np.sign(diff)[repulse])[:, np.newaxis]
        return ys
