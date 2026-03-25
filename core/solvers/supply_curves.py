"""
Supply-curve (dynamic equilibrium) solver
==========================================

Wraps DynamicEquilibriumSolver inside the BaseSolver interface and
provides a helper to load supply curves from the database.
"""

from __future__ import annotations

import logging
from typing import Dict, List, Tuple

import numpy as np

from core.solvers.base import BaseSolver

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# DynamicEquilibriumSolver (moved from models/table/solver.py)
# ---------------------------------------------------------------------------


class DynamicEquilibriumSolver:
    """
    Iterative solver that resolves quantity–price dynamics via merit-order
    supply curves.

    The solver iterates between:
    1. Computing market prices from the current output quantities and the
       tiered supply curves (merit-order dispatch).
    2. Re-solving the Leontief system with those prices.

    Convergence is declared when the output vector changes by less than
    ``tol`` (Euclidean norm) between iterations.

    Parameters
    ----------
    A_mon : np.ndarray, shape (n, n)
        Monetary technical coefficient matrix.
    isic_map : dict
        Mapping from ISIC code → matrix index.
    supply_data : dict
        Mapping from ISIC code → list of tier dicts
        ``[{"cap": float, "price": float}, ...]``.
    """

    def __init__(
        self,
        A_mon: np.ndarray,
        isic_map: Dict[str, int],
        supply_data: Dict[str, List[Dict]],
    ):
        self.A_mon = A_mon
        self.isic_map = isic_map
        self.n = len(isic_map)
        self.supply_tiers = self._vectorize_supply_data(supply_data)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _vectorize_supply_data(
        self, supply_data: Dict[str, List[Dict]]
    ) -> Dict[int, Tuple[np.ndarray, np.ndarray, np.ndarray]]:
        """Convert JSON tier lists to NumPy arrays for fast look-up."""
        vectorized: Dict[int, Tuple] = {}

        for isic, tiers in supply_data.items():
            if isic not in self.isic_map:
                continue
            idx = self.isic_map[isic]
            n_tiers = len(tiers)
            caps   = np.zeros(n_tiers)
            prices = np.zeros(n_tiers)

            for i, t in enumerate(tiers):
                c = t.get("cap")
                caps[i] = np.inf if (c is None or c == -1) else float(c)
                prices[i] = float(t.get("price", 0.0))

            bounds = np.concatenate(([0.0], np.cumsum(caps)[:-1]))
            vectorized[idx] = (bounds, caps, prices)

        return vectorized

    def get_market_prices(self, current_output: np.ndarray) -> np.ndarray:
        """Derive sector prices from the merit-order supply curves."""
        current_prices = np.zeros(self.n)

        for i in range(self.n):
            if i not in self.supply_tiers:
                current_prices[i] = 1.0
                continue

            bounds, caps, prices = self.supply_tiers[i]
            demand = current_output[i]

            if demand <= 0:
                current_prices[i] = prices[0]
                continue

            amounts = np.clip(demand - bounds, 0, caps)
            current_prices[i] = np.dot(amounts, prices) / demand

        return current_prices

    def update_value_matrix(self, current_prices: np.ndarray) -> np.ndarray:
        """Return the monetary A matrix (unchanged in the current formulation)."""
        return self.A_mon.copy()

    def solve(
        self,
        final_demand: np.ndarray,
        max_iter: int = 50,
        tol: float = 1e-3,
        verbose: bool = True,
    ) -> Dict:
        """
        Run the iterative solver.

        Returns
        -------
        dict
            ``{"status": "converged"|"failed", "output": ..., "prices": ...,
              "A_matrix": ..., "iterations": ...}``
        """
        current_output = np.array(final_demand, dtype=float)
        current_prices = np.zeros(self.n)

        print(f"Starting Solver for {self.n} sectors…")
        if verbose:
            print(f"\nInitial State:")
            print(f"  Final Demand:        {final_demand}")
            print(f"  Initial Output Guess: {current_output}\n")

        for iteration in range(max_iter):
            prev_output = current_output.copy()
            current_prices = self.get_market_prices(current_output)

            if verbose and iteration < 10:
                print(f"Iteration {iteration + 1}:")
                print(f"  Prices: {np.round(current_prices, 2)}")

            A_monetary = self.update_value_matrix(current_prices)

            if verbose and iteration < 3:
                print(f"  A_monetary:\n  {np.round(A_monetary, 4)}")

            I_minus_A = np.eye(self.n) - A_monetary

            if verbose and iteration == 0:
                col_sums = A_monetary.sum(axis=0)
                print(f"  A_monetary column sums: {np.round(col_sums, 4)}")
                bad_cols = np.where(col_sums >= 1.0)[0]
                if bad_cols.size:
                    print(
                        f"  WARNING: Columns {bad_cols} have sum >= 1.0 (non-productive!)"
                    )

            try:
                L_inv = np.linalg.inv(I_minus_A)
            except np.linalg.LinAlgError:
                logger.error("Matrix became singular!")
                print(f"  [X] SINGULAR MATRIX at iteration {iteration + 1}")
                return None  # type: ignore[return-value]

            current_output = L_inv @ final_demand

            if verbose and iteration < 10:
                print(f"  New Output: {np.round(current_output, 2)}")
                print(f"  Change: {np.linalg.norm(current_output - prev_output):.6f}\n")

            diff = np.linalg.norm(current_output - prev_output)
            if diff < tol:
                print(
                    f"[OK] Converged in {iteration + 1} iterations "
                    f"(change={diff:.6f} < tolerance={tol})"
                )
                return {
                    "status": "converged",
                    "output": current_output,
                    "prices": current_prices,
                    "A_matrix": A_monetary,
                    "iterations": iteration + 1,
                }

        print(f"[X] Max iterations ({max_iter}) reached without convergence.")
        return {
            "status": "failed",
            "output": current_output,
            "prices": current_prices,
            "iterations": max_iter,
        }


# ---------------------------------------------------------------------------
# SupplyCurveSolver — BaseSolver wrapper
# ---------------------------------------------------------------------------


class SupplyCurveSolver(BaseSolver):
    """
    Wraps :class:`DynamicEquilibriumSolver` in the :class:`BaseSolver` interface.

    Parameters
    ----------
    A_matrix : np.ndarray
        Technical coefficient matrix.
    isic_map : dict
        ISIC → index mapping.
    supply_data : dict
        ISIC → tier list.
    """

    def __init__(
        self,
        A_matrix: np.ndarray,
        isic_map: Dict[str, int],
        supply_data: Dict[str, List[Dict]],
    ):
        self._inner = DynamicEquilibriumSolver(A_matrix, isic_map, supply_data)

    def solve(self, demand: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Delegate to the DynamicEquilibriumSolver."""
        result = self._inner.solve(demand, verbose=False)
        if not result or result["status"] != "converged":
            raise RuntimeError("SupplyCurveSolver failed to converge")
        return result["output"], result["prices"]


# ---------------------------------------------------------------------------
# Database loader helper
# ---------------------------------------------------------------------------


def load_supply_curves_from_db(isic_map: Dict[str, int]) -> Dict[str, List[Dict]]:
    """
    Load tiered supply curve data from the database for all sectors in *isic_map*.

    Parameters
    ----------
    isic_map : dict
        ISIC → index mapping (the same one used to build A_matrix).

    Returns
    -------
    dict
        Mapping from ISIC code → tier list.
    """
    from db.repositories.supply_curve_repo import SupplyCurveDatabase

    scdb = SupplyCurveDatabase()
    all_curves = scdb.get_all_supply_curves()

    supply_data: Dict[str, List[Dict]] = {}
    for good in all_curves:
        if good.isic not in isic_map:
            continue
        if good.id_number == 9999:  # Foreign Exchange — skip
            continue
        tiers = good.price
        if not tiers:
            logger.warning(f"Sector {good.isic} has no price tiers — using defaults.")
            tiers = [{"cap": -1, "price": 1.0}]
        supply_data[good.isic] = tiers

    return supply_data
