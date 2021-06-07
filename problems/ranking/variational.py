
import numpy as np
from .fassolver import IlpSolver


class IL:
    def __init__(self, computer, fas_solver=None):
        self.computer = computer
        self.solver = fas_solver
        self.is_ilp = type(fas_solver) == IlpSolver

    def train(self, x_train, S_train, **kwargs):
        self.computer.set_support(x_train)
        self.computer.train(**kwargs)
        self.phi_init = S_train.astype(np.float)
        self.const = self.phi_init

    def __call__(self, x, tol=1e-3, solver=None, verbose=False):
        if solver is None:
            if self.solver is None:
                raise ValueError('FAS solver has not been specified.')
            solver = self.solver
        alpha = self.computer(x)
        # Because \ell(y, z) = - \phi(y)^\top \phi(z):
        alpha *= -1

        pred = np.empty((len(x), self.phi_init.shape[-1]), dtype=np.float)
        phi_pl = np.empty(self.phi_init.shape, dtype=np.float)

        for i in range(len(pred)):
            # To stabilize CPLEX
            alpha[i] /= np.abs(alpha[i]).max()
            alpha[i] *= 1e3
            # ------------------
            self.solve(alpha[i], pred[i], phi_pl, tol, solver)
            if verbose and not (100 * i) % len(x):
                print(i, end=', ')
        return pred

    def solve(self, alpha, out, phi_pl, tol, solver):
#         warmstart = [[] for i in range(len(phi_pl))]
        is_ilp = type(solver) == IlpSolver

        phi_pl[:] = self.phi_init[:]
        if self.is_ilp:
            self.solver.set_objective(alpha @ phi_pl)
            out[:] = self.solver.solve()
        else:
            self.solver.solve_out(alpha @ phi_pl, out)
        old_out = np.zeros(out.shape, dtype=np.float)

        # Alternate minimization
        while np.abs(out - old_out).max() > tol:
            old_out[:] = out[:]

            # Minimization of (y_i)_i
            if is_ilp:
                solver.set_objective(out)
                for j in range(len(phi_pl)):
                    if alpha[j] > 0:
                        solver.set_constraints(self.const[j])
#                         if len(warmstart[j]):
#                             solver.set_warmstart(warmstart[j])
                        phi_pl[j] = solver.solve()
#                         warmstart[j] = solver.get_warmstart()
                    else:
                        phi_pl[j] = 0

                out *= -1
                solver.set_objective(out)
                for j in range(len(phi_pl)):
                    if alpha[j] < 0:
                        solver.set_constraints(self.const[j])
#                         if len(warmstart[j]):
#                             solver.set_warmstart(warmstart[j])
                        phi_pl[j] = solver.solve()
#                         warmstart[j] = solver.get_warmstart()
            else:
                pre_sol_pos = solver.pre_solve(out)
                out *= -1
                pre_sol_neg = solver.pre_solve(out)
                for j in range(len(phi_pl)):
                    if alpha[j] > 0:
                        solver.incorporate_const_out(pre_sol_pos, self.const[j], phi_pl[j])
                    elif alpha[j] < 0:
                        solver.incorporate_const_out(pre_sol_neg, self.const[j], phi_pl[j])
                    else:
                        phi_pl[j] = 0

            # Minimization over z
            if self.is_ilp:
                self.solver.reset_constraints()
                self.solver.set_objective(alpha @ phi_pl)
                out[:] = self.solver.solve()
            else:
                self.solver.solve_out(alpha @ phi_pl, out)


class AC:
    def __init__(self, computer, fas_solver):
        self.computer = computer
        self.solver = fas_solver
        self.is_ilp = type(fas_solver) == IlpSolver

    def train(self, x_train, S_train, lambd, num=1, K_inv=None):
        self.computer.set_support(x_train)
        self.computer.train(lambd=lambd)
        phi = self.get_center(S_train, num, self.solver)
        self.computer.set_phi(phi)

    def __call__(self, x, verbose=False):
        c = self.computer.call_with_phi(x)
        c *= -1
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

    @staticmethod
    def get_center(S_train, num, solver):
        is_ilp = type(solver) == IlpSolver

        phi = np.empty(S_train.shape)
        tmp = np.empty((num, S_train.shape[1]))
        for i in range(len(phi)):
            tmp[:] = 0
            ctl = num
            for j in range(num):
                c = np.random.randn(S_train.shape[1])
                if is_ilp:
                    solver.set_constraints(S_train[i])
                    solver.set_objective(c)
                    tmp[j] = solver.solve()
                else:
                    tmp[j] += solver.solve_const(c, S_train[i])
                if j:
                    if (tmp[:j] == tmp[j]).mean(axis=1).max() == 1:
                        tmp[j] = 0
                        ctl -= 1
            phi[i] = tmp.sum(axis=0) / ctl

        if is_ilp:
            solver.reset_constraints()

        return phi


class SP:
    def __init__(self, computer, fas_solver):
        self.computer = computer
        self.solver = fas_solver
        self.is_ilp = type(fas_solver) == IlpSolver

    def train(self, x_train, S_train, **kwargs):
        self.computer.set_support(x_train)
        self.computer.train(**kwargs)
        self.phi_init = S_train.astype(np.float)
        self.const = self.phi_init

    def __call__(self, x, tol=1e-10, solver=None, verbose=False):
        if solver is None:
            if self.solver is None:
                raise ValueError('FAS solver has not been specified.')
            solver = self.solver
        alpha = self.computer(x)
        alpha *= -1

        pred = np.empty((len(x), self.phi_init.shape[-1]), dtype=np.float)
        phi_pl = np.empty(self.phi_init.shape, dtype=np.float)

        for i in range(len(pred)):
            self.solve(alpha[i], pred[i], phi_pl, tol, solver)
            if verbose and not (100 * i) % len(x):
                print(i, end=", ")
        return pred

    def solve(self, alpha, out, phi_pl, tol, solver):
#         warmstart = [[] for i in range(len(phi_pl))]
        is_ilp = type(solver) == IlpSolver

        phi_pl[:] = self.phi_init[:]
        if self.is_ilp:
            self.solver.set_objective(alpha @ phi_pl)
            out[:] = self.solver.solve()
        else:
            self.solver.solve_out(alpha @ phi_pl, out)
        old_out = np.zeros(out.shape, dtype=np.float)

        # Alternate minimization
        ctl = 0
        while np.abs(out - old_out).max() > tol and ctl < 100:
            ctl += 1
            old_out[:] = out[:]

            # Minimization of (y_i)_i
            if is_ilp:
                solver.set_objective(out)
                for j in range(len(phi_pl)):
                    if alpha[j] < 0:
                        solver.set_constraints(self.const[j])
#                         if len(warmstart[j]):
#                             solver.set_warmstart(warmstart[j])
                        phi_pl[j] = solver.solve()
#                         warmstart[j] = solver.get_warmstart()
                    else:
                        phi_pl[j] = 0

                out *= -1
                solver.set_objective(out)
                for j in range(len(phi_pl)):
                    if alpha[j] > 0:
                        solver.set_constraints(self.const[j])
#                         if len(warmstart[j]):
#                             solver.set_warmstart(warmstart[j])
                        phi_pl[j] = solver.solve()
#                         warmstart[j] = solver.get_warmstart()
            else:
                pre_sol_pos = solver.pre_solve(out)
                out *= -1
                pre_sol_neg = solver.pre_solve(out)
                for j in range(len(phi_pl)):
                    if alpha[j] < 0:
                        solver.incorporate_const_out(pre_sol_pos, self.const[j], phi_pl[j])
                    elif alpha[j] > 0:
                        solver.incorporate_const_out(pre_sol_neg, self.const[j], phi_pl[j])
                    else:
                        phi_pl[j] = 0

            # Minimization over z
            if self.is_ilp:
                self.solver.reset_constraints()
                self.solver.set_objective(alpha @ phi_pl)
                out[:] = self.solver.solve()
            else:
                self.solver.solve_out(alpha @ phi_pl, out)
