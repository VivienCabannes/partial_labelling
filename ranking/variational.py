
import numpy as np
from .fassolver import IlpSolver


class IL:
    def __init__(self, kernel, fas_solver):
        self.kernel = kernel
        self.solver = fas_solver
        self.is_ilp = type(fas_solver) == IlpSolver

    def train(self, x_train, S_train, lambd, K_inv=None):

        if K_inv is None:
            n_train = len(S_train)
            self.kernel.set_support(x_train)
            K = self.kernel.get_k()
            w, v = np.linalg.eigh(K)
            w += n_train * lambd
            w **= -1
            self.K_inv = (v * w) @ v.T
            self.K_inv *= -1
        else:
            self.K_inv = K_inv
        
        self.phi_init = S_train.astype(np.float)
        self.const = self.phi_init

    def __call__(self, x, tol=1e-3, solver=None, verbose=False):
        if solver is None:
            solver = self.solver
        K_x = self.kernel(x).T
        alpha = K_x @ self.K_inv            
        
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


class SP:
    def __init__(self, kernel, fas_solver):
        self.kernel = kernel
        self.solver = fas_solver
        self.is_ilp = type(fas_solver) == IlpSolver

    def train(self, x_train, S_train, lambd, K_inv=None):

        if K_inv is None:
            n_train = len(S_train)
            self.kernel.set_support(x_train)
            K = self.kernel.get_k()
            w, v = np.linalg.eigh(K)
            w += n_train * lambd
            w **= -1
            self.K_inv = (v * w) @ v.T
            self.K_inv *= -1
        else:
            self.K_inv = K_inv
        
        self.phi_init = S_train.astype(np.float)
        self.const = self.phi_init

    def __call__(self, x, tol=1e-10, solver=None, verbose=False):
        if solver is None:
            solver = self.solver
        K_x = self.kernel(x).T
        alpha = K_x @ self.K_inv            
        
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
