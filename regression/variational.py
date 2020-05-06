
import numpy as np

    
class IL:
    def __init__(self, kernel, projection):
        self.kernel = kernel
        self.projection = projection
        
    def train(self, x_train, S_train, lambd):
        n_train = len(x_train)
        
        self.kernel.set_support(x_train)
        K = self.kernel.get_k()
        w, v = np.linalg.eigh(K)
        self.K_inv = v / (w + n_train * lambd) @ v.T

        self.S_train = S_train

    def __call__(self, x, tol=1e-6, max_it=1e4):
        alpha = self.kernel(x).T @ self.K_inv
        alpha_norm = alpha / alpha.sum(axis=1)[:, np.newaxis]
        return self.disambiguation(alpha_norm, self.S_train, tol, max_it, self.projection)
    
    @staticmethod
    def disambiguation(alpha_norm, S_train, tol, max_it, projection):
        y_init = S_train[0]
        z = np.empty((len(alpha_norm), y_init.shape[1]), dtype=y_init.dtype)
        
        for i in range(len(alpha_norm)):
            y = y_init.copy()
            y_old = np.zeros(y.shape)
            it = 0
            while np.max(np.abs(y - y_old)) > tol and it < max_it:
                it += 1
                y_old[:] = y[:]
                z[i] = alpha_norm[i] @ y
                y = projection(z[i], alpha_norm[i], S_train)
            z[i] = alpha_norm[i] @ y
            if not i % 10:
                print(i, end=',')
        return z
    
    
def projection_il_ir(z, alpha_norm, Ss):
    cs, rs = Ss
    ys = np.full(cs.shape, z)
    diff = (cs - z).squeeze() 
    outside = np.abs(diff) > rs
    ys[outside] = cs[outside] - (rs[outside] * np.sign(diff)[outside])[:, np.newaxis]
    repulse = alpha_norm < 0
    ys[repulse] = cs[repulse] + (rs[repulse] * np.sign(diff)[repulse])[:, np.newaxis]
    return ys