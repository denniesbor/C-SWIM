"""
Social Accounting Matrix (SAM) Balancer Module

This module implements a Social Accounting Matrix (SAM) balancing algorithm 
using constrained optimization techniques as described in Hosoe, Gasawa, and 
Hashimoto's "Textbook of Computable General Equilibrium Modelling" (2010).

The balancing procedure uses a weighted least-squares approach to adjust values 
in an unbalanced SAM while preserving economic structure. It solves the following 
constrained minimization problem:

    minimize ∑∑ ((x_i,j - x⁰_i,j)/x⁰_i,j)²
     x_i,j   i j

    subject to:
    ∑ x_i,j = ∑ x_j,i  ∀i
    j        j

Where:
- x_i,j: Value in i-th row and j-th column in the adjusted SAM
- x⁰_i,j: Value in i-th row and j-th column in the original SAM
- i,j: Row and column labels in the SAM

This objective function minimizes the sum of squared deviation rates of adjusted 
cell values from original ones, analogous to weighted least-squares estimation.
The constraint ensures the row sum equals column sum for each account, preserving 
the fundamental accounting balance property of SAMs.

The implementation uses Pyomo to formulate and solve this nonlinear optimization
problem, with additional handling for structural zeros and numerical stability.

Functions:
---------
build_sam_balancer(sam, epsilon=1e-10)
    Build a Pyomo optimization model for the SAM balancing problem
    
solve_sam_balancer(model, solver='ipopt', tee=True)
    Solve the optimization model using a specified solver
    
extract_balanced_sam(model, original_sam, epsilon=1e-10)
    Extract the balanced SAM from the solved model
    
balance_sam(sam, epsilon=1e-10, decimal_places=1, solver='ipopt', tee=True)
    Complete balancing pipeline from unbalanced SAM to balanced result

References:
----------
Hosoe, N., Gasawa, K., & Hashimoto, H. (2010). Textbook of Computable General 
Equilibrium Modelling: Programming and Simulations. Palgrave Macmillan, pp. 69-71.
"""

import numpy as np
import pandas as pd
import pyomo.environ as pyo
from pyomo.opt import SolverFactory


def build_sam_balancer(sam: pd.DataFrame, epsilon=1e-10):
    """
    Build a Pyomo optimization model for balancing a Social Accounting Matrix.
    
    Parameters:
    -----------
    sam : pd.DataFrame
        The input Social Accounting Matrix to be balanced
    epsilon : float, optional
        Small value to prevent division by zero, default is 1e-10
        
    Returns:
    --------
    m : pyo.ConcreteModel
        Pyomo optimization model ready to be solved
    """
    I = list(sam.index)                 # account labels
    n = len(I)
    X0 = sam.to_numpy(dtype=float)
    
    # Binary mask: 1 if original cell is non-zero, 0 if structural zero
    mask = (X0 != 0).astype(float)
    
    m = pyo.ConcreteModel()
    m.I = pyo.RangeSet(0, n-1)
    m.J = pyo.RangeSet(0, n-1)
    
    # Decision variables (non-negative by default)
    m.x = pyo.Var(m.I, m.J, domain=pyo.NonNegativeReals)
    
    # Zero-cell constraints
    def zero_rule(_, i, j):
        return m.x[i, j] == 0 if mask[i, j] == 0 else pyo.Constraint.Skip
    m.zero_fix = pyo.Constraint(m.I, m.J, rule=zero_rule)
    
    # Row = column constraints
    def balance_rule(_, k):
        return sum(m.x[k, j] for j in m.J) == sum(m.x[i, k] for i in m.I)
    m.balance = pyo.Constraint(m.I, rule=balance_rule)
    
    # Objective: weighted least-squares with protection against division by zero
    def obj_rule(_):
        return sum(mask[i, j] * 
                   ((m.x[i, j] - X0[i, j]) / (X0[i, j] + epsilon))**2 
                   if X0[i, j] != 0 else 
                   (m.x[i, j])**2  # Alternative penalty for originally-zero cells that aren't structural zeros
                   for i in m.I for j in m.J)
    m.obj = pyo.Objective(rule=obj_rule, sense=pyo.minimize)
    
    return m


def solve_sam_balancer(model, solver='ipopt', tee=True):
    """
    Solve the SAM balancing optimization model.
    
    Parameters:
    -----------
    model : pyo.ConcreteModel
        Pyomo model built with build_sam_balancer
    solver : str, optional
        Name of the solver to use, default is 'ipopt'
    tee : bool, optional
        Whether to display solver output, default is True
        
    Returns:
    --------
    results : SolverResults
        Solver results object
    """
    opt = SolverFactory(solver)
    results = opt.solve(model, tee=tee)
    return results


def extract_balanced_sam(model, original_sam, epsilon=1e-10):
    """
    Extract the balanced SAM from the solved model.
    
    Parameters:
    -----------
    model : pyo.ConcreteModel
        Solved Pyomo model
    original_sam : pd.DataFrame
        Original SAM dataframe (needed for index/column names)
    epsilon : float, optional
        Threshold below which values are set to zero, default is 1e-10
        
    Returns:
    --------
    balanced_sam : pd.DataFrame
        Balanced Social Accounting Matrix
    """
    # Extract values from the model
    balanced_array = np.array([[pyo.value(model.x[i, j]) 
                                for j in model.J] for i in model.I])
    
    # Create a mask for values below threshold
    zero_mask = np.abs(balanced_array) < epsilon
    
    # Apply the mask to set small values to exactly zero
    balanced_array[zero_mask] = 0.0
    
    # Create the DataFrame with cleaned values
    balanced_sam = pd.DataFrame(balanced_array,
                                index=original_sam.index,
                                columns=original_sam.columns)
    
    return balanced_sam


def balance_sam(sam, epsilon=1e-10, decimal_places=1, solver='ipopt', tee=True):
    """
    Complete SAM balancing pipeline: build model, solve, and extract results.
    
    Parameters:
    -----------
    sam : pd.DataFrame
        Original Social Accounting Matrix to balance
    epsilon : float, optional
        Small value to prevent division by zero, default is 1e-10
    decimal_places : int, optional
        Number of decimal places to round results to, default is 6
    solver : str, optional
        Name of the solver to use, default is 'ipopt'
    tee : bool, optional
        Whether to display solver output, default is True
        
    Returns:
    --------
    balanced_sam : pd.DataFrame
        Balanced Social Accounting Matrix
    """
    # Build the model
    model = build_sam_balancer(sam, epsilon=epsilon)
    
    # Solve the model
    results = solve_sam_balancer(model, solver=solver, tee=tee)
    
    # Extract and format the balanced SAM
    balanced_sam = extract_balanced_sam(model, sam, epsilon=epsilon)
    
    # Round to specified decimal places
    if decimal_places is not None:
        balanced_sam = balanced_sam.round(decimal_places)
    
    return balanced_sam