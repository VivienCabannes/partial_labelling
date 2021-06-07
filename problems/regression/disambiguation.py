
import numpy as np


class DF:
    def __init__(self, computer, projection):
        self.computer = computer
        self.projection = projection

    def train(self, x_train, S_train, tol=1e-6, max_it=1e10, **kwargs):
        self.computer.set_support(x_train)
        self.computer.train(**kwargs)
        alpha = self.computer(x_train)
        phi = self.disambiguation(alpha, S_train, tol, max_it, self.projection)
        self.computer.set_phi(phi)

    def __call__(self, x):
        return self.computer.call_with_phi(x)

    @staticmethod
    def disambiguation(alpha, S_train, tol, max_it, projection):
        y_df = S_train[0].copy()
        y_old = np.zeros(y_df.shape)
        it = 0
        while np.max(np.abs(y_df - y_old)) > tol and it < max_it:
            it += 1
            y_old[:] = y_df[:]
            z_df = alpha @ y_df
            y_df = projection(alpha @ z_df, S_train)
        return y_df

    @staticmethod
    def projection_ir(ys, Ss):
        cs, rs = Ss
        diff = (ys - cs).squeeze()
        outside = np.abs(diff) > rs
        ys[outside] = cs[outside] + (rs[outside] * np.sign(diff)[outside])[:, np.newaxis]
        return ys

    @staticmethod
    def projection_pr(ys, Ss):
        cs, rs, bad_ind = Ss
        ind = ys.squeeze() < 0
        ind &= bad_ind
        ys[ind] *= -1
        diff = (ys - cs).squeeze()
        outside = np.abs(diff) > rs
        ys[outside] = cs[outside] + (rs[outside] * np.sign(diff)[outside])[:, np.newaxis]
        ys[ind] *= -1
        return ys
