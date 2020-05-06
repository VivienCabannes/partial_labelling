
import numpy as np


class DF:
    def __init__(self, kernel, projection):
        self.kernel = kernel
        self.projection = projection

    def train(self, x_train, S_train, lambd, tol=1e-6, max_it=1e10):
        n_train = len(x_train)
        
        self.kernel.set_support(x_train)
        K = self.kernel.get_k()
        w, v = np.linalg.eigh(K)
        self.K_inv = v / (w + n_train * lambd) @ v.T

        w_reg = w / (w + n_train * lambd)
        alpha = (w_reg * v) @ v.T
        self.y_df = self.disambiguation(alpha, S_train, tol, max_it, self.projection)

    def __call__(self, x):
        alpha = self.kernel(x).T @ self.K_inv
        alpha_norm = alpha / alpha.sum(axis=1)[:, np.newaxis]
        return alpha_norm @ self.y_df

    @staticmethod
    def disambiguation(alpha, S_train, tol, max_it, projection):
        alpha_norm = alpha / alpha.sum(axis=1)[:, np.newaxis]

        y_df = S_train[0].copy()
        y_old = np.zeros(y_df.shape)
        it = 0
        while np.max(np.abs(y_df - y_old)) > tol and it < max_it:
            it += 1
            y_old[:] = y_df[:]
            z_df = alpha_norm @ y_df
            y_df = projection(alpha_norm @ z_df, S_train)
        return y_df


def projection_df_ir(ys, Ss):
    cs, rs = Ss
    diff = (ys - cs).squeeze()
    outside = np.abs(diff) > rs
    ys[outside] = cs[outside] + (rs[outside] * np.sign(diff)[outside])[:, np.newaxis]
    return ys