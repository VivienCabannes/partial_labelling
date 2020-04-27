
import numpy as np


class DF:
    def __init__(self, kernel):
        self.kernel = kernel

    def train(self, x_train, S_train, lambd, quadratic=False, method='FW', nb_epochs=1):
        n_train, self.m = S_train.shape

        self.kernel.set_support(x_train)
        K = self.kernel.get_k()
        w, v = np.linalg.eigh(K)
        w_reg = w / (w + n_train * lambd)
        alpha = (w_reg * v) @ v.T

        if quadratic:
            alpha = alpha.T @ alpha
            y_train = self.quadratic_disambiguation(alpha, S_train, method, nb_epochs)
        else:
            y_train = self.disambiguation(alpha, S_train)

        self.beta = v / (w + n_train * lambd) @ (v.T @ y_train)
        self.y_train = y_train

    def __call__(self, x):
        out = self.kernel(x).T @ self.beta
        aux = np.tile(np.arange(self.m), (x.shape[0], 1))
        out[:] = aux == out.argmax(axis=1)[:, np.newaxis]
        return out

    @staticmethod
    def disambiguation(alpha, S_train):
        n_train, m = S_train.shape
        forbidden = np.invert(S_train)

        y_train = S_train.astype(np.float)
        y_train /= y_train.sum(axis=1)[:, np.newaxis]

        aux_argmax = np.tile(np.arange(m), (n_train, 1))
        z = np.zeros(y_train.shape)
        z_old = np.ones(y_train.shape)
        while not (z == z_old).all():
            z_old[:] = z[:]

            np.matmul(alpha, y_train, out=z)
            z[:] = aux_argmax == z.argmax(axis=1)[:, np.newaxis]

            np.matmul(alpha, z, out=y_train)
            y_train[forbidden] = -np.infty
            y_train[:] = aux_argmax == y_train.argmax(axis=1)[:, np.newaxis]

        return y_train
        
    @staticmethod
    def quadratic_disambiguation(alpha, S_train, method, nb_epochs):
        n_train, m = S_train.shape
        forbidden = np.invert(S_train)

        y_train = S_train.astype(np.float)
        y_train /= y_train.sum(axis=1)[:, np.newaxis]

#         alpha -= np.eye(n_train)
        aux_argmax = np.tile(np.arange(m), (n_train, 1))

        if method.lower() == 'bw':
            # Blockwise Frank-Wolfe        
            for t in range(nb_epochs):
                i = np.random.randint(n_train)
                
                score = alpha[i] @ y_train
                score[forbidden[i]] = -np.infty
                dir_bw = np.argmax(score)
                
                y_train[i] *= t / (2*n_train + t)
                y_train[i, dir_bw] += 2*n_train / (2*n_train + t)
    
        else:
            # Frank-Wolfe
            for t in range(nb_epochs):
                dir_fw = alpha @ y_train
                dir_fw[forbidden] = -np.infty
                dir_fw[:] = aux_argmax == dir_fw.argmax(axis=1)[:, np.newaxis]

                dir_fw -= y_train
                dir_fw *= 2 / (t + 2)
                y_train += dir_fw

#         y_train = aux_argmax == y_train.argmax(axis=1)[:, np.newaxis]                
        return y_train
