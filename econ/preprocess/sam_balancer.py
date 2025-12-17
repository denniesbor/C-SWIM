"""
Social Accounting Matrix (SAM) Balancer Module
Authors: Dennies and Oguthon

This module implements a Social Accounting Matrix (SAM) balancing algorithm
using constrained optimization techniques. The balancing procedure uses a
weighted least-squares approach to adjust values in an unbalanced SAM while
preserving economic structure.

The algorithm minimizes the sum of squared deviation rates subject to the
constraint that row sums equal column sums for each account, preserving
the fundamental accounting balance property of SAMs.

"""

import numpy as np
import pandas as pd
import pyomo.environ as pyo
from pyomo.opt import SolverFactory


def build_sam_balancer(sam: pd.DataFrame, epsilon=1e-10):
    """Build a Pyomo optimization model for balancing a Social Accounting Matrix."""
    I = list(sam.index)
    n = len(I)
    X0 = sam.to_numpy(dtype=float)

    mask = (X0 != 0).astype(float)

    m = pyo.ConcreteModel()
    m.I = pyo.RangeSet(0, n - 1)
    m.J = pyo.RangeSet(0, n - 1)

    m.x = pyo.Var(m.I, m.J, domain=pyo.NonNegativeReals)

    for i in range(n):
        for j in range(n):
            if X0[i, j] > 0.0:
                m.x[i, j].setlb(1e-10)

    ACT_NAMES = [
        "AGR",
        "MINING",
        "UTIL_CONST",
        "MANUF",
        "TRADE_TRANSP",
        "INFO",
        "FIRE",
        "PROF_OTHER",
        "EDUC_ENT",
        "G",
    ]
    ACT = [I.index(s) for s in ACT_NAMES if s in I]
    CAP_i, LAB_i, IDT_i = I.index("CAP"), I.index("LAB"), I.index("IDT")
    VA0 = {j: X0[CAP_i, j] + X0[LAB_i, j] + X0[IDT_i, j] for j in ACT}

    def va_lb_rule(_, j):
        return (m.x[CAP_i, j] + m.x[LAB_i, j] + m.x[IDT_i, j]) >= 0.95 * VA0[j]

    m.va_lb = pyo.Constraint(ACT, rule=va_lb_rule)

    def zero_rule(_, i, j):
        return m.x[i, j] == 0 if mask[i, j] == 0 else pyo.Constraint.Skip

    m.zero_fix = pyo.Constraint(m.I, m.J, rule=zero_rule)

    def balance_rule(_, k):
        return sum(m.x[k, j] for j in m.J) == sum(m.x[i, k] for i in m.I)

    m.balance = pyo.Constraint(m.I, rule=balance_rule)

    def obj_rule(_):
        return sum(
            (
                mask[i, j] * ((m.x[i, j] - X0[i, j]) / (X0[i, j] + epsilon)) ** 2
                if X0[i, j] != 0
                else (m.x[i, j]) ** 2
            )
            for i in m.I
            for j in m.J
        )

    m.obj = pyo.Objective(rule=obj_rule, sense=pyo.minimize)

    return m


def solve_sam_balancer(model, solver="ipopt", tee=True):
    """Solve the SAM balancing optimization model."""
    opt = SolverFactory(solver)
    results = opt.solve(model, tee=tee)
    return results


def extract_balanced_sam(model, original_sam, epsilon=1e-10):
    """Extract the balanced SAM from the solved model."""
    balanced_array = np.array(
        [[pyo.value(model.x[i, j]) for j in model.J] for i in model.I]
    )

    zero_mask = np.abs(balanced_array) < epsilon
    balanced_array[zero_mask] = 0.0

    balanced_sam = pd.DataFrame(
        balanced_array, index=original_sam.index, columns=original_sam.columns
    )

    return balanced_sam


def balance_sam(sam, epsilon=1e-10, decimal_places=1, solver="ipopt", tee=True):
    """Complete SAM balancing pipeline: build model, solve, and extract results."""
    model = build_sam_balancer(sam, epsilon=epsilon)
    results = solve_sam_balancer(model, solver=solver, tee=tee)
    balanced_sam = extract_balanced_sam(model, sam, epsilon=epsilon)

    if decimal_places is not None:
        balanced_sam = balanced_sam.round(decimal_places)

    return balanced_sam