
import numpy as np


class FoldsGenerator:
    def __init__(self, x, y, S, nb_folds=10):
        self.x = x
        self.y = y
        self.S = S
        nb_data = x.shape[0]
        if nb_folds > 1:
            self.get_cross_folds(nb_data, nb_folds)
        self.max_fold = nb_folds
        self.fold = 0

    def get_cross_folds(self, nb_data, nb_folds):
        ind = np.random.permutation(np.arange(nb_data))
        self.cross_ind = np.zeros((nb_folds, nb_data), dtype=np.bool_)

        inc = nb_data / nb_folds
        for i in range(nb_folds):
            ind_test = ind >= (i * inc)
            ind_test &= ind < ((i + 1) * inc)
            self.cross_ind[i, :] = ind_test.copy()

    def __call__(self):
        if self.max_fold == 1:
            return (self.x, self.S), (None, None)
        if self.fold == self.max_fold:
            return (None, None), (None, None)
        else:
            ind_test = self.cross_ind[self.fold]
            ind_train = np.invert(ind_test)
            x_train = self.x[ind_train]
            S_train = self.S[ind_train]
            x_test = self.x[ind_test]
            y_test = self.y[ind_test]
            self.fold += 1
            return (x_train, S_train), (x_test, y_test)

    def reset(self):
        self.fold = 0