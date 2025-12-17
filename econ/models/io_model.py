"""
Leontief Input-Output Analysis Module

Authors: Dennies, Oguthon

This module implements a Leontief Input-Output (IO) model for economic impact analysis
of exogenous shocks. It provides functionality to analyze how changes in final demand
or value-added components propagate through the interdependent sectors of an economy.

The implementation supports both demand-driven analysis (using the Leontief inverse)
and supply-driven analysis (using the Ghosh inverse), allowing for comprehensive
assessment of backward and forward linkages in the production network.

Features:
- Load and process standard IO data (A-matrix, output vectors)
- Calculate Leontief inverse for demand-side multiplier effects
- Calculate Ghosh inverse for supply-side effects
- Simulate impacts of final demand shocks across sectors
- Simulate impacts of value-added shocks (e.g., labor or capital constraints)
- Calculate sectoral output multipliers

The module is designed to work with the 10-sector aggregation produced by the
production technology matrix generator, bridging input-output analysis with more
complex CGE modeling frameworks.

Technical Background:
The Leontief model is based on the fundamental equation:
    X = AX + F
where X is the gross output vector, A is the direct requirements matrix,
and F is the final demand vector. Solving for X:
    X = (I-A)^(-1)F
where (I-A)^(-1) is the Leontief inverse matrix (L).

For supply-side analysis, the Ghosh model uses the allocation coefficients:
    B = X^(-1)AX
and the Ghosh inverse:
    G = (I-B)^(-1)

Both models assume constant returns to scale, fixed input proportions,
and no supply constraints in the standard formulation.

References:
Miller, R. E., & Blair, P. D. (2009). Input-Output Analysis: Foundations and Extensions.
Cambridge University Press.
"""

from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
from numpy.linalg import solve

from configs import setup_logger, get_data_dir

logger = setup_logger("InputOutputModel")
DATA_LOC = get_data_dir()


class InputOutputModel:
    """
    Leontief IO wrapper for 10-sector data set.

    Expects four files in `DATA_LOC / 10sector`:
        ├── direct_requirements.csv   # 10×10 A-matrix
        ├── gross_output.csv          # 10×1 column, header = any year label
        ├── final_demand.csv          # 10×k (optional, read if present)
        └── value_added.csv           # m×10 (optional, read if present)
    """

    def __init__(
        self,
        data_path: str | Path,
        *,
        final_demand: Optional[pd.Series] = None,
        value_added: Optional[pd.Series] = None,
    ):
        """Initialize the Input-Output Model."""
        self.path = DATA_LOC / data_path
        self._load_A()
        self._load_X()
        self._make_identities()
        self.set_exogenous(final_demand, value_added)
        self._Linv = self._Ginv = None

    def _load_A(self) -> None:
        """Load the direct requirements matrix (A-matrix)."""
        a_df = pd.read_csv(self.path / "direct_requirements.csv", index_col=0)
        if a_df.shape[0] != a_df.shape[1]:
            raise ValueError("A-matrix must be square")
        self.A = a_df.to_numpy(float)
        self.sectors = a_df.index.tolist()

    def _load_X(self) -> None:
        """Load the gross output vector."""
        x_df = pd.read_csv(self.path / "gross_output.csv", index_col=0)
        self.X = x_df.iloc[:, 0].to_numpy(float)
        if len(self.X) != len(self.sectors):
            raise ValueError("Gross-output length ≠ A-matrix size")

    def _make_identities(self):
        """Create identity matrix and inverse of gross output diagonal matrix."""
        n = len(self.sectors)
        self.I = np.eye(n)
        self._X_hat_inv = np.diag(1 / self.X)

    def set_exogenous(
        self,
        final_demand: Optional[pd.Series],
        value_added: Optional[pd.Series],
    ):
        """Set exogenous variables for the model."""
        if final_demand is None and (self.path / "final_demand.csv").exists():
            fd_df = pd.read_csv(self.path / "final_demand.csv", index_col=0)
            final_demand = fd_df.sum(axis=1)

        if value_added is None and (self.path / "value_added.csv").exists():
            va_df = pd.read_csv(self.path / "value_added.csv", index_col=0)
            value_added = va_df.sum(axis=0)

        self.final_demand = (
            final_demand.to_numpy(float) if final_demand is not None else None
        )
        self.value_added = (
            value_added.to_numpy(float) if value_added is not None else None
        )

    @property
    def L(self):
        """Leontief inverse matrix: (I-A)^(-1)."""
        if self._Linv is None:
            self._Linv = solve(self.I - self.A, self.I)
        return self._Linv

    @property
    def G(self):
        """Ghosh inverse matrix: (I-B)^(-1), where B = X^(-1) * A * X."""
        if self._Ginv is None:
            B = self._X_hat_inv @ self.A @ np.diag(self.X)
            self._Ginv = solve(self.I - B, self.I)
        return self._Ginv

    def total_output_from_final_demand(self, delta_fd: pd.Series) -> pd.Series:
        """Calculate the impact on total output from a change in final demand."""
        out = self.L @ delta_fd.reindex(self.sectors).to_numpy(float)
        return pd.Series(out, index=self.sectors, name="ΔX_fd")

    def total_output_from_value_added(self, delta_va: pd.Series) -> pd.Series:
        """Calculate the impact on total output from a change in value added."""
        out = delta_va.reindex(self.sectors).to_numpy(float) @ self.G
        return pd.Series(out, index=self.sectors, name="ΔX_va")

    def output_multipliers(self):
        """Calculate output multipliers for each sector."""
        return pd.Series(self.L.sum(axis=0), index=self.sectors, name="multiplier")


if __name__ == "__main__":
    io = InputOutputModel("10sector")

    d_fd = pd.Series(0, index=io.sectors)
    d_fd["MANUF"] = -10
    dx_fd = io.total_output_from_final_demand(d_fd)

    d_va = pd.Series(0, index=io.sectors)
    d_va["UTIL_CONST"] = -5
    dx_va = io.total_output_from_value_added(d_va)

    logger.info("ΔX from final-demand shock\n", dx_fd.round(2))
    logger.info("\nΔX from value-added shock\n", dx_va.round(2))
