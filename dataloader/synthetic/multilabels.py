
import numpy as np


class MLSynthesizer:
#     nb_class = 4
    nb_class = 6

    def __init__(self, n_train):
        self.n_train = n_train

    def f(self, x):
        if len(x.shape) < 2:
            x = x[np.newaxis, :]
        y = np.zeros((x.shape[0], self.nb_class), dtype=np.bool_)
#         y[:, 0] = x[:, 0] < 1
#         y[:, 1] = x[:, 0] > 2
#         y[:, 2] = x[:, 1] < 1
#         y[:, 3] = x[:, 1] > 2
        y[:, 0] = x[:, 0] < 1
        y[:, 1] = x[:, 0] >= 1
        y[:, 1] &= x[:, 0] <= 2
        y[:, 2] = x[:, 0] > 2
        y[:, 3] = x[:, 1] < 1
        y[:, 4] = x[:, 1] >= 1
        y[:, 4] &= x[:, 1] <= 2
        y[:, 5] = x[:, 1] > 2
        return 2 * y.astype(np.float) - 1

    def get_trainset(self):
        x_train = 3 * np.random.rand(self.n_train, 2)
        y_train = self.f(x_train)
        return x_train, y_train

    def get_testset(self, n_test):
        num = int(np.sqrt(n_test))
        mesh = np.meshgrid(np.linspace(0, 3, num),
                           np.linspace(0, 3, num))
        x_test = np.empty((num**2, 2), dtype=np.float64)
        x_test[:, 0] = mesh[0].flatten()
        x_test[:, 1] = mesh[1].flatten()
        return x_test, self.f(x_test)

    @staticmethod
    def synthetic_corruption(labels, corruption_rate, skewed=False):
        S_train = labels.astype(np.float)
        lost = np.random.rand(*labels.shape) < corruption_rate
        S_train[lost] = 0
        if skewed:
            S_train[S_train < 0] = 0
        return S_train
