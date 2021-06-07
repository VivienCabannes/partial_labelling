
import numba
import numpy as np


@numba.jit("i8[:, :](i8)", nopython=True)
def canonical_map(m):
    ind_map = np.full((m, m), m**2, dtype=np.int64)
    ind = 0
    for j in range(m):
        for i in range(j):
            ind_map[i, j] = ind
            ind += 1
    return ind_map


def get_emb(scores, ind_map):
    m = len(scores)
    m_emb = (m*(m-1))//2
    emb = np.empty(m_emb, dtype=np.float64)
    if scores.dtype == np.int64:
        fill_emb_i8(scores, emb, ind_map)
    else:
        fill_emb_f8(scores, emb, ind_map)
    return emb


@numba.jit("(i8[:], f8[:], i8[:, :])", nopython=True)
def fill_emb_i8(scores, emb, ind_map):
    m = len(ind_map)
    for j in range(m):
        for i in range(j):
            emb[ind_map[i, j]] = scores[i] > scores[j]
    emb *= 2
    emb -= 1


@numba.jit("(f8[:], f8[:], i8[:, :])", nopython=True)
def fill_emb_f8(scores, emb, ind_map):
    m = len(ind_map)
    for j in range(m):
        for i in range(j):
            emb[ind_map[i, j]] = scores[i] > scores[j]
    emb *= 2
    emb -= 1


def get_emb_from_rank(rank, ind_map):
    m = len(ind_map)
    m_emb = (m*(m-1)) // 2
    emb = np.zeros(m_emb, dtype=np.float64)
    fill_emb_from_rank(rank, emb, ind_map)
    return emb


@numba.jit("(i8[:], f8[:], i8[:, :])", nopython=True)
def fill_emb_from_rank(rank, emb, ind_map):
    for i_, i in enumerate(rank):
        for j in rank[i_+1:]:
            if i < j:
                ind = ind_map[i, j]
                emb[ind] = -1
            if j < i:
                ind = ind_map[j, i]
                emb[ind] = 1


def get_sym_emb(emb, ind_map):
    m = len(ind_map)
    sym_emb = np.zeros((m, m), dtype=np.float64)
    fill_sym_emb(emb, sym_emb, ind_map)
    return sym_emb


@numba.jit("(f8[:], f8[:, :], i8[:, :])", nopython=True)
def fill_sym_emb(emb, sym_emb, ind_map):
    m = len(ind_map)
    for j in range(m):
        sym_emb[j, j] = 0
        for i in range(j):
            ind = ind_map[i, j]
            sym_emb[i, j] = emb[ind]
            sym_emb[j, i] = -emb[ind]


def get_sym_embs(embs, ind_map):
    m = len(ind_map)
    sym_embs = np.zeros((len(embs), m, m), dtype=np.float64)
    fill_sym_embs(embs, sym_embs, ind_map)
    return sym_embs


@numba.jit("(f8[:, :], f8[:, :, :], i8[:, :])")
def fill_sym_embs(embs, sym_embs, ind_map):
    for i in range(len(embs)):
        fill_sym_emb(embs[i], sym_embs[i], ind_map)
