
import cplex as cp
import numba
import numpy as np


class IlpSolver:
    def __init__(self, ind_map, method='primal'):
        self.ind_map = ind_map

        # Instanciate LP
        self.solver = cp.Cplex()
        self.instanciate_variables()
        self.set_transitivity_constraints()

        self.choose_solver_method(method)
        self.info_for_reset = None

    def set_constraints(self, const):
        """
        const is of shape (m_emb,), with:
           - c[ind_map[i,j]] = 1, means that x_ij = 1
           - c[ind_map[i,j]] = -1, means that x_ij = -1
           - c[ind_map[i,j]] = 0, means that x_ij is not constrained
        """
        self.reset_constraints()
        m_emb = len(const)

        # get equality constraints
        ind = const != 0
        if not (ind).sum():
            return
        index = np.arange(m_emb)[ind].tolist()
        values = const[ind].tolist()

        # cplex formatting
        cp_const = [(index[i], values[i]) for i in range(len(index))]
        self.solver.variables.set_lower_bounds(cp_const)
        self.solver.variables.set_upper_bounds(cp_const)

        self.info_for_reset = index

    def set_objective(self, c):
        self.solver.objective.set_linear(enumerate(c))

    def solve(self):
        self.solver.solve()
        return np.array(self.solver.solution.get_values())

    def get_warmstart(self):
        basis = self.solver.solution.basis.get_basis()
        return basis

    def set_warmstart(self, basis):
        self.solver.start.set_start(
            col_status=basis[0], row_status=basis[1],
            col_primal=[], row_primal=[], col_dual=[], row_dual=[])

    def export(self, file_name):
        self.solver.write(file_name)

    def import_pb(self, file_name):
        self.solver.read(file_name)

    def shut_up(self):
        self.solver.set_results_stream(None)
        self.solver.set_warning_stream(None)
        self.solver.set_error_stream(None)
        self.solver.set_log_stream(None)

    def delete(self):
        self.solver.end()

    def choose_solver_method(self, solver_method):
        """
        solver method should be:
            - 'auto' for automatic
            - 'primal' = primal simplex (default)
            - 'dual' = dual simplex
            - 'network' for network simplex
            - 'barrier' for barrier
            - 'sifting' for sifting
            - 'concurrent' for concurrent optimizers
        """
        i = getattr(self.solver.parameters.lpmethod.values, solver_method)
        self.solver.parameters.lpmethod.set(i)

        # For gradient, the devex pricing is adapated to the type of problem we are solving
        dgradient = 'full' # 'full', 'devex',...
        i = getattr(self.solver.parameters.simplex.dgradient.values, dgradient)
        self.solver.parameters.simplex.dgradient.set(i)

        pgradient = 'steep' # 'partial', 'devex', 'steep',...
        i = getattr(self.solver.parameters.simplex.pgradient.values, pgradient)
        self.solver.parameters.simplex.pgradient.set(i)

    def reset_constraints(self):
        index = self.info_for_reset
        if index is None:
            return

        cp_const = [(index[i], -1) for i in range(len(index))]
        self.solver.variables.set_lower_bounds(cp_const)

        cp_const = [(index[i], 1) for i in range(len(index))]
        self.solver.variables.set_upper_bounds(cp_const)

        self.info_for_reset = None

    def instanciate_variables(self):
        m = len(self.ind_map)
        m_emb = (m*(m-1)) // 2
        self.solver.variables.add(ub=[1.0] * m_emb, lb=[-1.0] * m_emb)

    @staticmethod
    @numba.jit("(i8[:, :], i8[:, :])", nopython=True)
    def _fill_tr_const(tr_const, ind_map):
        m = len(ind_map)
        ind = 0
        for k in range(m):
            for j in range(k):
                for i in range(j):
                    tr_const[ind, 0] = ind_map[i, j]
                    tr_const[ind, 1] = ind_map[j, k]
                    tr_const[ind, 2] = ind_map[i, k]
                    ind += 1

    def set_transitivity_constraints(self):
        m = len(self.ind_map)
        nb_const = ((m * (m + 1) * (2*m+1)) // 6 - 3 * (m * (m-1)) // 2 - m) // 2
        tr_const = np.empty((nb_const, 3), dtype=np.int)

        self._fill_tr_const(tr_const, self.ind_map)

        # Formatting for cplex
        cp_const = [[i, [1.0, 1.0, -1.0]] for i in tr_const.tolist()]

        # x_ij + x_jk - x_ik <= 1
        self.solver.linear_constraints.add(lin_expr=cp_const,
                                           senses='L' * len(cp_const),
                                           rhs=[1.0] * len(cp_const))
        # x_ij + x_jk - x_ik >= -1
        self.solver.linear_constraints.add(lin_expr=cp_const,
                                           senses='G' * len(cp_const),
                                           rhs=[-1.0] * len(cp_const))
