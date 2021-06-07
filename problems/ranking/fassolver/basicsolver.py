
import numba
import numpy as np
from .embedding import fill_sym_emb, fill_emb_f8, fill_emb_from_rank
from .engine import FasSolver


@numba.jit('(i8, f8[::1], i8[::1], b1[::1], b1[:,::1], i8[::1])', nopython=True)
def _insert_const(item, scores, rank_pl, visited, const_pl, range_pl):
    if visited[item]:
        return
    ind = const_pl[item] & (~visited)
    sm_items = range_pl[ind]
    sm_items = sm_items[np.argsort(scores[sm_items])]

    for i in sm_items:
        _insert_const(i, scores, rank_pl, visited, const_pl, range_pl)

    rank_pl[visited.sum()] = item
    visited[item] = True


@numba.jit('(i8[::1], f8[::1], f8[::1], f8[::1], b1[::1], i8[::1],'\
            'f8[:,::1], b1[:,::1], i8[:,::1], i8[::1])', nopython=True)
def _incorporate_const_out(rank, scores, const, out, visited, rank_pl, 
                           sym_pl, const_pl, ind_map, range_pl):
    fill_sym_emb(const, sym_pl, ind_map)
    for i in range_pl:
        for j in range_pl:
            const_pl[i,j] = sym_pl[i, j] == 1

    visited[:] = False
    for item in rank:
        _insert_const(item, scores, rank_pl, visited, const_pl, range_pl)
    fill_emb_from_rank(rank_pl, out, ind_map)


class BasicFasSolver(FasSolver):
    def __init__(self, ind_map):
        self.ind_map = ind_map
        self.IL_met = 'pre_solve'

        # Placeholders
        m = len(ind_map)
        self.sym_pl = np.empty((m, m), dtype=np.float)
        self.score_pl = np.empty(m, dtype=np.float)
        self.const_pl = np.empty((m, m), dtype=np.bool_)
        self.range_pl = np.arange(m)
        self.rank_pl = np.empty(m, dtype=np.int)
        self.visited = np.empty(m, dtype=np.bool_)

    def solve_out(self, c, out):
        fill_sym_emb(c, self.sym_pl, self.ind_map)
        np.sum(self.sym_pl, axis=1, out=self.score_pl) 
        self.score_pl *= -1
        fill_emb_f8(self.score_pl, out, self.ind_map)

    def solve_const_out(self, c, const, out):
        pre_sol = self.pre_solve(c)
        self.incorporate_const_out(pre_sol, const, out)

    def pre_solve(self, c):
        """
        First pass to solve the minimum feedback arc set problem reading:
           argmin_e <e, c>
        Subject to constraint not defined yet.

        This function allows to solve efficiently a big number of 
        instance with the same objective but different constraint.
        This is useful for the infimum loss.
        """
        fill_sym_emb(c, self.sym_pl, self.ind_map)
        scores = np.sum(self.sym_pl, axis=1)
        scores *= -1
        rank = scores.argsort()
        return rank, scores

    def incorporate_const_out(self, pre_sol, const, out):
        rank, scores = pre_sol
        _incorporate_const_out(rank, scores, const, out, 
            self.visited, self.rank_pl, self.sym_pl, 
            self.const_pl, self.ind_map, self.range_pl)

#     def define_const(self, const):
#         raise NotImplementedError

#     def resolve_out(self, c, out):
#         raise NotImplementedError
