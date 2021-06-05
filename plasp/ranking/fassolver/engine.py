
import numpy as np


class FasSolver:
    def __init__(self, ind_map):
        self.ind_map = ind_map
        self.IL_met = '' # either pre_solve, either resolve

    def solve(self, c):
        """
        Solve the minimum feedback arc set problem reading:
           argmin_e <e, c>
        Subject to:
           - e Kendall's embedding of a permutation
        """
        emb = np.empty(c.shape, dtype=np.float)
        self.solve_out(c, emb)
        return emb

    def solve_const(self, c, const):
        """
        Solve the minimum feedback arc set problem reading:
           argmin_e <e, c>
        Subject to:
           - e[const != 0] = const
           - e Kendall's embedding of a permutation
        """
        emb = np.empty(c.shape, dtype=np.float)
        self.solve_const_out(c, const, emb)
        return emb

    def pre_solve(self, c):
        """
        First pass to solve the minimum feedback arc set problem reading:
           argmin_e <e, c>
        Subject to constraint not defined yet.

        This function allows to solve efficiently a big number of 
        instance with the same objective but different constraint.
        This is useful for the infimum loss.
        """
        raise NotImplementedError

    def incorporate_const(self, pre_sol, const):
        """
        Use pre solution of the problem argmin_e <e, c> and retune it to solve:
           argmin_e <e, c>
        Subject to:
           - e[const != 0] = const
           - e Kendall's embedding of a permutatiom

        This function allows to solve efficiently a big number of 
        instance with the same objective but different constraint.
        This is useful for the infimum loss.
        """
        emb = np.empty(c.shape, dtype=np.float)
        self.incorporate_const_out(pre_sol, const, emb)
        return emb

    def define_const(self, const):
        """
        Define constraint for the following ``resolve`` call.

        Useful for the disambiguation framework with warmstartable call.
        """
        raise NotImplementedError

    def resolve(self, c):
        """
        Solve by retaking last solution calculation:
           argmin_e <e, c>
        Subject to:
           - e[const != 0] = const
           - e Kendall's embedding of a permutation
        where const has been define by ``define_const``.

        Useful for the disambiguation framework with warmstartable call.
        """
        emb = np.empty(c.shape, dtype=np.float)
        self.resolve_out(c, emb)
        return emb

    def solve_out(self, c, out):
        raise NotImplementedError

    def solve_const_out(self, c, const, out):
        raise NotImplementedError

#     def pre_solve(self, c):
#         raise NotImplementedError

    def incorporate_const_out(self, pre_sol, const, out):
        raise NotImplementedError

#     def define_const(self, const):
#         raise NotImplementedError

    def resolve_out(self, c, out):
        raise NotImplementedError
