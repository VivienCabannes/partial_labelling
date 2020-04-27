
import numpy as np


class DF:
    def __init__(self, kernel):
        self.kernel = kernel

    def train(self, x_train, S_train, lambd, nb_epochs, epsilon=0, method='FW', verbose=False):
        n_train = len(x_train)
        method = method.lower()
        if not method in ['bw', 'fw']:
            raise ValueError('"' + method.upper() + '" is nor "BW" (blockwise FW), nor "FW" (Frank-Wolfe.')

        self.kernel.set_support(x_train)
        K = self.kernel.get_k()
        w, v = np.linalg.eigh(K)
        w += n_train * lambd
        w **= -1
        K_inv = (v * w) @ v.T
        
        phi = self.disambiguation(K_inv, S_train, nb_epochs, method, epsilon)
        
        self.beta = K_inv @ phi
        
        if verbose:
            return phi
        
    def __call__(self, x):
        K_x = self.kernel(x).T
        return K_x @ self.beta
    
    def thres_pred(self, x, threshold):
        pred = self(x)
        pred -= threshold
        np.sign(pred, out=pred)
        return pred
    
    def topk_pred(self, x, k):
        soft_pred = self(x)
        sort_pred = np.argsort(soft_pred, axis=1)[:, -k:]
        pred = np.zeros(soft_pred.shape, dtype=np.bool_)
        self.fill_topk_pred(pred, sort_pred)
        return pred
            
    @classmethod
    def disambiguation(cls, K_inv, S_train, nb_epochs, method, epsilon):
        # initialization
        n_train = len(S_train)
        phi = np.asfortranarray(S_train, dtype=np.float) # make sure phi_i are contiguous in memory in Phi.

        const_idx = phi != 0
        change_idx = ~const_idx

        if method == 'bw':
            # Blockwise Frank-Wolfe
            for t in range(nb_epochs):
                i = np.random.randint(n_train)

                dir_BWFW = K_inv[i] @ phi
                dir_BWFW *= -1
                if epsilon:
                    dir_BWFW -= epsilon / (2*n_train)
                np.sign(dir_BWFW, out=dir_BWFW)
                dir_BWFW *= 2*n_train / (2*n_train + t)

                phi[i, change_idx[i]] *= t / (2*n_train + t)
                phi[i, change_idx[i]] += dir_BWFW[change_idx[i]]

        else:
            # Frank-Wolfe
            for t in range(nb_epochs):
                dir_FW = K_inv @ phi
                dir_FW *= -1
                if epsilon:
                    dir_FW -= epsilon / (2*n_train)
                np.sign(dir_FW, out=dir_FW)

                dir_FW -= phi
                dir_FW *= 2 / (t + 2)

                dir_FW[const_idx] = 0

                phi += dir_FW

        return phi
    
    
class DF:
    def __init__(self, kernel):
        self.kernel = kernel

    def train(self, x_train, S_train, lambd, threshold, quadratic=False, method='FW', nb_epochs=100):
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
            y_train = self.disambiguation(alpha, S_train, threshold)

        self.beta = v / (w + n_train * lambd) @ (v.T @ y_train)
        self.y_train = y_train

    def __call__(self, x):
        out = self.kernel(x).T @ self.beta
        aux = np.tile(np.arange(self.m), (x.shape[0], 1))
        out[:] = aux == out.argmax(axis=1)[:, np.newaxis]
        return out

    @staticmethod
    def disambiguation(alpha, S_train, threshold):
        n_train, m = S_train.shape
        y_train = S_train.astype(np.float)

        z = np.zeros(y_train.shape)
        z_old = np.ones(y_train.shape)
        while not (z == z_old).all():
            z_old[:] = z[:]

            np.matmul(alpha, y_train, out=z)
            z[:] = z > threshold

            np.matmul(alpha, z, out=y_train)
            y_train[S_train] = threshold
            y_train[:] = y_train >= threshold

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
