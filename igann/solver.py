"""Bounded-Variable Least-Squares algorithm."""
import numpy as np
import torch
from torch.linalg import lstsq
from torch import optim
import torch.nn as nn


def compute_kkt_optimality_torch(g, on_bound):
    """Compute the maximum violation of KKT conditions."""
    g_kkt = g * on_bound
    free_set = on_bound == 0
    g_kkt[free_set] = torch.abs(g[free_set])
    return torch.max(g_kkt)

def bvls_torch(A, b, x_lsq, lb, ub, tol, max_iter):
    '''
    Translated from scipy/optimize/_lsq/bvls.py into pytorch
    '''
    m, n = A.shape

    x = x_lsq.clone()
    on_bound = torch.zeros(n)

    mask = x < lb
    x[mask] = lb[mask]
    on_bound[mask] = -1

    mask = x > ub
    x[mask] = ub[mask]
    on_bound[mask] = 1

    free_set = on_bound == 0
    active_set = ~free_set.bool()
    free_set, = torch.where(free_set)

    r = A.matmul(x) - b
    cost = 0.5 * torch.dot(r, r)
    initial_cost = cost
    g = A.T.matmul(r)

    cost_change = None
    step_norm = None
    iteration = 0

    while free_set.size(0) > 0:
        optimality = compute_kkt_optimality_torch(g, on_bound)
        print(f'{iteration}; {cost}; {cost_change}; {step_norm}; {optimality}')

        iteration += 1
        x_free_old = x[free_set].clone()

        A_free = A[:, free_set]
        b_free = b - A.matmul(x * active_set)
        z, _, _, _ = lstsq(A_free, b_free)

        lbv = z < lb[free_set]
        ubv = z > ub[free_set]
        v = lbv | ubv

        if torch.any(lbv):
            ind = free_set[lbv]
            x[ind] = lb[ind]
            active_set[ind] = True
            on_bound[ind] = -1

        if torch.any(ubv):
            ind = free_set[ubv]
            x[ind] = ub[ind]
            active_set[ind] = True
            on_bound[ind] = 1

        ind = free_set[~v]
        x[ind] = z[~v]

        r = A.matmul(x) - b
        cost_new = 0.5 * torch.dot(r, r)
        cost_change = cost - cost_new
        cost = cost_new
        g = A.T.matmul(r)
        step_norm = torch.linalg.norm(x[free_set] - x_free_old)

        if torch.any(v):
            free_set = free_set[~v]
        else:
            break

    if max_iter is None:
        max_iter = n
    max_iter += iteration

    termination_status = None

    # Main BVLS loop.

    optimality = compute_kkt_optimality_torch(g, on_bound)
    for iteration in range(iteration, max_iter):
        print(f'{iteration}; {cost}; {cost_change}; {step_norm}; {optimality}')

        if optimality < tol:
            termination_status = 1

        if termination_status is not None:
            break

        move_to_free = torch.argmax(g * on_bound)
        on_bound[move_to_free] = 0
        free_set = on_bound == 0
        active_set = ~free_set.bool()
        free_set, = torch.where(free_set)

        x_free = x[free_set]
        x_free_old = x_free.clone()
        lb_free = lb[free_set]
        ub_free = ub[free_set]

        A_free = A[:, free_set]
        b_free = b - A.matmul(x * active_set)
        z, _, _, _ = lstsq(A_free, b_free)

        lbv, = torch.where(z < lb_free)
        ubv, = torch.where(z > ub_free)
        v = torch.cat((lbv, ubv))

        if v.size(0) > 0:
            alphas = torch.cat((
                lb_free[lbv] - x_free[lbv],
                ub_free[ubv] - x_free[ubv])) / (z[v] - x_free[v])

            i = torch.argmin(alphas)
            i_free = v[i]
            alpha = alphas[i]

            x_free = x_free * (1 - alpha)
            x_free += alpha * z

            if i < lbv.size(0):
                on_bound[free_set[i_free]] = -1
            else:
                on_bound[free_set[i_free]] = 1
        else:
            x_free = z

        x[free_set] = x_free
        step_norm = torch.linalg.norm(x_free - x_free_old)

        r = A.matmul(x) - b
        cost_new = 0.5 * torch.dot(r, r)
        cost_change = cost - cost_new

        if cost_change < tol * cost:
            termination_status = 2
        cost = cost_new

        g = A.T.matmul(r)
        optimality = compute_kkt_optimality_torch(g, on_bound)

    if termination_status is None:
        termination_status = 0

    return x, termination_status, cost, optimality


A = np.random.randn(1000,10)
b = np.random.randn(1000)

x_lsq = np.linalg.lstsq(A, b, rcond=None)[0]

lb = np.zeros(10)
ub = np.ones(10) * 10

A = torch.from_numpy(A).float()
b = torch.from_numpy(b).float()

x_lsq = torch.from_numpy(x_lsq).float()
lb = torch.from_numpy(lb).float()
ub = torch.from_numpy(ub).float()

tol = 1e-3
max_iter = None

x_pt, termination_status_pt, cost_pt, optimality_pt = bvls_torch(A, b, x_lsq, lb, ub, tol, max_iter)