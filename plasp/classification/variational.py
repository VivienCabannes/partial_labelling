
import numpy as np


class Variational:
    def __init__(self, kernel, method):
        self.kernel = kernel
        self.method = method.lower()
        if not self.method in ['ac', 'il']:
            raise ValueError('"' + method + '" is nor "AC", nor "IL".')

    def train(self, x_train, S_train, lambd):
        n_train = len(x_train)

        if self.method == 'ac':
            phi = np.asfortranarray(S_train, dtype=np.float)
            phi /= phi.sum(axis=1)[:, np.newaxis]
        else:
            phi = np.asfortranarray(S_train, dtype=np.float)

        self.kernel.set_support(x_train)
        K_lambda = self.kernel.get_k()
        K_lambda += lambd * n_train * np.eye(n_train)
        self.beta = np.linalg.solve(K_lambda, phi)

    def __call__(self, x):
        K_x = self.kernel(x).T
        idx = (K_x @ self.beta).argmax(axis=1)
        return idx
        
    
class AC(Variational):
    def __init__(self, kernel):
        super(AC, self).__init__(kernel, 'ac') 

        
class IL(Variational):
    def __init__(self, kernel):
        super(IL, self).__init__(kernel, 'il') 