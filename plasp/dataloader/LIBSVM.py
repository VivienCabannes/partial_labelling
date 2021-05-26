
import numpy as np
from sklearn.datasets import load_svmlight_file
from .config import LIBSVM_DIR
from .engine import DataLoader


class LIBSVMLoader(DataLoader):
    problem = 'CL'
    data_path = LIBSVM_DIR
    datasets = [
        'acoustic',
        'aloi',
        'cifar10',
        'combined',
        'connect4',
        'covtype',
        'dna',
        'glass',
        'iris',
        'letter',
        'mnist',
        'news20',
        'pendigits',
        'poker',
        'protein',
        'satimage',
        'sector',
        'segment',
        'seismic',
        'sensorless',
        'shuttle',
        'SVHN',
        'svmguide2',
        'svmguide4',
        'usps',
        'vehicle',
        'vowel',
        'wine',
    ]

    def __init__(self, name, **kwargs):
        super().__init__(name, **kwargs)

    def read_trainset(self):
        path = self.data_path / self.name / (self.name)
        X, y = self.read_data(path)
        self.Y = np.unique(y)
        y_train = np.zeros((y.size, len(self.Y)), dtype=np.float64)
        for i, label in enumerate(self.Y):
            y_train[y == label, i] = 1
        return X, y_train

    def read_testset(self):
        path = self.data_path / self.name / (self.name + '.t')
        return self.read_data(path)

    def synthetic_corruption(self, y_train, corruption):
        S_train = y_train.astype(np.float)
        S_train += np.random.rand(*S_train.shape)
        S_train = S_train >= (1 - corruption)
        return S_train

    def skewed_corruption(self, y_train, corruption, index):
        S_train = y_train.astype(np.float)
        ind = S_train[:, index].astype(np.bool_)
        S_train[ind] += np.random.rand(ind.sum(), S_train.shape[-1])
        S_train = S_train >= (1 - corruption)
        return S_train

    @staticmethod
    def read_data(path):
        with open(path, 'rb') as f:
            X, y = load_svmlight_file(f)
        return X.toarray(), y
