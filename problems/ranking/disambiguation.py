
import numpy as np
from .fassolver import IlpSolver


class DF:
    def __init__(self, computer, fas_solver):
        self.computer = computer
        self.solver = fas_solver
        self.is_ilp = type(self.solver) == IlpSolver

    def train(self, x_train, S_train, threshold=1e-3, nb_epochs=1, solver=None, quadratic=False, method='FW', **kwargs):
        if solver is None:
            if self.solver is None:
                raise ValueError('FAS solver has not been specified.')
            solver = self.solver

        self.computer.set_support(x_train)
        self.computer.train(**kwargs)
        alpha = self.computer(x_train)

        if quadratic:
            alpha = alpha.T @ alpha
            phi = self.quadratic_disambiguation(alpha, S_train, method, nb_epochs, solver)
        else:
            phi = self.disambiguation(alpha, S_train, threshold, solver)

        self.computer.set_phi(phi)

    def __call__(self, x, verbose=False):
        out = self.computer.call_with_phi(x)
        out *= -1
        for i in range(len(x)):
            if self.is_ilp:
                # To stabilize CPLEX
                out[i] /= np.abs(out[i]).max()
                out[i] *= 1e3
                # ------------------
                self.solver.set_objective(out[i])
                out[i] = self.solver.solve()
            else:
                self.solver.solve_out(out[i], out[i])
            if verbose and not (100 * i) % len(x):
                print(i, end=", ")
        return out

    @staticmethod
    def disambiguation(alpha, S_train, threshold, solver):
        is_ilp = type(solver) == IlpSolver
        y_train = S_train.astype(np.float)
        const = S_train.astype(np.float)

        z = np.zeros(y_train.shape)
        z_old = np.ones(y_train.shape)
        while np.abs(z - z_old).max() > threshold:
            z_old[:] = z[:]

            # Minimization over z
            np.matmul(alpha, y_train, out=z)
            z *= -1
            for i in range(len(z)):
                if is_ilp:
                    solver.set_objective(z[i])
                    z[i] = solver.solve()
                else:
                    solver.solve_out(z[i], z[i])

            # Minimization over y
            np.matmul(alpha, z, out=y_train)
            y_train *= -1
            for j in range(len(y_train)):
                if is_ilp:
                    solver.set_constraints(const[j])
                    solver.set_objective(y_train[j])
                    y_train[j] = solver.solve()
                else:
                    solver.solve_const_out(y_train[j], const[j], y_train[j])

            if is_ilp:
                solver.reset_constraints()

        return y_train

    def quadratic_disambiguation(alpha, S_train, method, nb_epochs, solver):
        is_ilp = type(solver) == IlpSolver
        y_train = S_train.astype(np.float)
        const = S_train.astype(np.float)

#         alpha -= np.eye(n_train)
        if method.lower() == 'bw':
            # Blockwise Frank-Wolfe
            n_train = len(y_train)
            for t in range(nb_epochs):
                i = np.random.randint(n_train)

                if is_ilp:
                    solver.set_constraints(const[i])
                    solver.set_objective(-alpha[i] @ y_train)
                    dir_bw = solver.solve()
                else:
                    dir_bw = solver.solve_const(-alpha[i] @ y_train, const[i])

                y_train[i] *= t / (2*n_train + t)
                y_train[i] += (2*n_train / (2*n_train + t)) * dir_bw

        else:
            # Frank-Wolfe
            for t in range(nb_epochs):
                dir_fw = alpha @ y_train
                dir_fw *= -1
                for i in range(len(dir_fw)):
                    if is_ilp:
                        solver.set_constraints(const[i])
                        solver.set_objective(dir_fw[i])
                        dir_fw[j] = solver.solve()
                    else:
                        solver.solve_const_out(dir_fw[i], const[i], dir_fw[i])

                dir_fw -= y_train
                dir_fw *= 2 / (t + 2)
                y_train += dir_fw

        if is_ilp:
            solver.reset_constraints()
        return y_train
