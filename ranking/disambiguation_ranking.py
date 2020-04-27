
import numpy as np


class DF:
    def __init__(self, kernel, fas_solver):
        self.kernel = kernel
        self.solver = fas_solver

    def train(self, x_train, S_train, lambd, nb_epochs, solver=None, verbose=False):
        if solver is None:
            solver = self.solver
        n_train = len(S_train)

        self.kernel.set_support(x_train)
        K = self.kernel.get_k()
        w, v = np.linalg.eigh(K)
        w += n_train * lambd
        w **= -1
        K_inv = (v * w) @ v.T

        phi = self.disambiguation(K_inv, S_train, nb_epochs, solver, verbose)

        self.beta = K_inv @ phi
        self.beta *= -1
        
        if verbose:
            return phi

    def __call__(self, x, verbose=False):
        K_x = self.kernel(x).T
        c = K_x @ self.beta
        pred = np.empty(c.shape, dtype=np.float)
        for i in range(len(x)):
            self.solver.solve_out(c[i], pred[i])
            if verbose and not (100 * i) % len(x):
                print(i, end=", ")
        return pred

    def disambiguation(self, K_inv, S_train, nb_epochs, solver, verbose):
        # Allow one solver per points for warm start
        ctl = False
        if type(solver) is list:
            ctl = True
            seen = np.zeros(n_train, dtype=np.bool_)

        n_train = len(S_train)
        
        phi = S_train.astype(np.float)
        dir_BWFW = np.empty(phi.shape[-1], dtype=np.float)

        for t in range(nb_epochs):
            i = np.random.randint(n_train)

            grad = K_inv[i] @ phi
            if ctl:
                if not seen[i]:
                    solver[i].define_const(S_train[i])
                    seen[i] = True
                solver[i].resolve_out(grad, dir_BWFW)
            else:
                solver.solve_const_out(grad, S_train[i], dir_BWFW)

            dir_BWFW -= phi[i]
            dir_BWFW *= 2 * n_train / (t + 2 * n_train)
            phi[i] += dir_BWFW

            if verbose and not (100 * t) % nb_epochs:
                print(t, end=", ")
        
        return phi