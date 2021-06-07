
import numpy as np


class CLSynthesizer:
    Y = np.arange(9)

    def __init__(self, n_train):
        self.n_train = n_train

    def f(self, x):
        if len(x.shape) < 2:
            x = x[np.newaxis, :]
        y = x[:, 0].astype(np.int)
        y *= 3
        y += x[:, 1].astype(np.int)
        return y

    def get_trainset(self):
        x_train = 3 * np.random.rand(self.n_train, 2)
        y_train = self.f(x_train)
        return x_train, y_train

    def get_testset(self, n_test):
        num = int(np.sqrt(n_test))
        mesh = np.meshgrid(np.linspace(0, 3, num+1)[:-1],
                           np.linspace(0, 3, num+1)[:-1])
        x_test = np.empty((num**2, 2), dtype=np.float64)
        x_test[:, 0] = mesh[0].flatten()
        x_test[:, 1] = mesh[1].flatten()
        return x_test, self.f(x_test)

    def synthetic_corruption(self, labels, corruption_rate):
        pos_label = np.zeros((labels.size, len(self.Y)), dtype=np.float64)
        for i, label in enumerate(self.Y):
            pos_label[labels == label, i] = 1
        pos_label += np.random.rand(*pos_label.shape)
        S_train = pos_label >= (1 - corruption_rate)
        return S_train
