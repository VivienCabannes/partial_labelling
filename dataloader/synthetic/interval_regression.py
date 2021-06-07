
import numpy as np


class IRSynthesizer:
    omega = 10

    def __init__(self, n_train):
        self.n_train = n_train

    def f(self, x):
        y = np.sin(self.omega * x)
        return y

    def get_trainset(self):
        x_train = np.random.rand(self.n_train)[:, np.newaxis]
        y_train = self.f(x_train)
        return x_train, y_train

    def get_testset(self, n_test):
        x_test = np.linspace(0, 1, n_test)[:, np.newaxis]
        y_test = self.f(x_test)
        return x_test, y_test

    def synthetic_corruption(self, labels, corruption, skewed=False):
        exp_par, offset = corruption
        r_train = abs(offset) - np.log(np.random.rand(self.n_train)) / exp_par
        if skewed:
            mu = np.random.rand(self.n_train) * r_train
            c_train = labels + np.abs(mu)[:, np.newaxis] * np.sign(labels)
        else:
            mu = (2 * np.random.rand(self.n_train) - 1) * r_train
            c_train = labels + mu[:, np.newaxis]
        return c_train, r_train
