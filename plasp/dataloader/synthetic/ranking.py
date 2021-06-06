
import numba
import numpy as np


# See ranking.fassolver.embedding.canonical_map
@numba.jit("i8[:, :](i8)", nopython=True)
def canonical_map(m):
    ind_map = np.full((m, m), m**2, dtype=np.int64)
    ind = 0
    for j in range(m):
        for i in range(j):
            ind_map[i, j] = ind
            ind += 1
    return ind_map


class RKSynthesizer:
    def __init__(self, m):
        self.m = m
        self.m_emb = (m*(m-1))//2
        self.ind_map = canonical_map(m)
        self.instanciate()

    def instanciate(self):
        a = np.random.randn(self.m)
        b = np.random.randn(self.m)
        if self.m == 4:
            a[:] = [.5, 0, 1, -.75]
            b[:] = [.75, .5, .5, .75]
        self.a = a[:, np.newaxis]
        self.b = b[:, np.newaxis]

    def f(self, x, verbose=False):
        y_score = self.a @ x[np.newaxis, :] + self.b
        y = np.zeros((len(x), self.m_emb), dtype=np.float64)
        self.fill_y(y, y_score, self.ind_map) 
        if verbose:
            return y, y_score
        return y

    def get_trainset(self, n_train, verbose=False):
        x_train = np.random.rand(n_train)
        x_train *= 2
        x_train -= 1
        y_train = self.f(x_train, verbose=verbose)
        return x_train, y_train

    def get_testset(self, n_test, verbose=False):
        x_test = np.linspace(-1, 1, n_test)
        y_test = self.f(x_test, verbose=verbose)
        return x_test, y_test

#     @staticmethod
#     def synthetic_corruption(y_train, corruption_rate):
#         S_train = y_train.copy()
#         S_train[np.random.rand(*S_train.shape) < corruption_rate] = 0
#         return S_train

    def synthetic_corruption(self, y_train, corruption_rate, skewed=False, y_score=None):
        S_train = y_train.copy()
        if not skewed:
            S_train[np.random.rand(*S_train.shape) < corruption_rate] = 0
        else:
            dist = np.empty(S_train.shape)
            self.fill_distance(dist, y_score, self.ind_map)
            np.abs(dist, out=dist)
            dist /= np.max(dist, axis=1)[:, np.newaxis]
            ind_lost = dist > 1 - corruption_rate
#             ind_lost = dist < corruption_rate
            S_train[ind_lost] = 0
        return S_train

    @staticmethod
    @numba.jit("(f8[:, :], f8[:, :], i8[:, :])", nopython=True)
    def fill_y(y_train, y_score, ind_map):
        m = len(ind_map)
        for j in range(m):
            for i in range(j):
                ind = ind_map[i, j]
                y_train[:,ind] = 2 * (y_score[i] > y_score[j]) - 1

    @staticmethod
    @numba.jit("(f8[:, :], f8[:, :], i8[:, :])", nopython=True)
    def fill_distance(dist, y_score, ind_map):
        m = len(ind_map)
        for j in range(m):
            for i in range(j):
                ind = ind_map[i, j]
                dist[:,ind] = y_score[i] - y_score[j]
