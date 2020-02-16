
import numpy as np
from .fassolver import IlpSolver


class AC:
    def __init__(self, kernel, fas_solver):
        self.kernel = kernel
        self.solver = fas_solver
        self.is_ilp = type(fas_solver) == IlpSolver

    def train(self, x_train, S_train, lambd, num=1, K_inv=None):
        phi = np.empty(S_train.shape)
        tmp = np.empty((num, S_train.shape[1]))
        for i in range(len(phi)):
            tmp[:] = 0
            ctl = num
            for j in range(num):
                c = np.random.randn(S_train.shape[1])
                if self.is_ilp:
                    self.solver.set_constraints(S_train[i])
                    self.solver.set_objective(c)
                    tmp[j] = self.solver.solve()
                else:
                    tmp[j] += self.solver.solve_const(c, S_train[i])
                if j:
                    if (tmp[:j] == tmp[j]).mean(axis=1).max() == 1:
                        tmp[j] = 0
                        ctl -= 1
            phi[i] = tmp.sum(axis=0) / ctl        
    
        if self.is_ilp:
            self.solver.reset_constraints()

        if K_inv is None:        
            n_train = len(x_train)
            self.kernel.set_support(x_train)
            K_lambda = self.kernel.get_k()
            K_lambda += lambd * n_train * np.eye(n_train)
            self.beta = np.linalg.solve(K_lambda, phi)
            self.beta *= -1
        else:
            self.beta = K_inv @ phi
        
    def __call__(self, x, verbose=False):
        K_x = self.kernel(x).T
        c = K_x @ self.beta
        pred = np.empty(c.shape, dtype=np.float)
        for i in range(len(x)):
            if self.is_ilp:
                self.solver.set_objective(c[i])
                pred[i] = self.solver.solve()
            else:
                self.solver.solve_out(c[i], pred[i])
            if verbose and not (100 * i) % len(x):
                print(i, end=", ")
        return pred