"""
Leontief inverse solver
========================

Implements the classical Leontief X = (I - A)^-1 F solution.
"""

from __future__ import annotations

import logging
from typing import Tuple

import numpy as np

from core.solvers.base import BaseSolver

logger = logging.getLogger(__name__)


class LeontiefSolver(BaseSolver):
    """
    Solves the I-O system using the Leontief inverse.

    X = (I - A)^{-1} F

    Prices are fixed at 1.0 for all sectors.

    Parameters
    ----------
    A_matrix : np.ndarray, shape (n, n)
        Technical coefficient matrix.
    """

    def __init__(self, A_matrix: np.ndarray):
        n = A_matrix.shape[0]
        try:
            self._L = np.linalg.inv(np.eye(n) - A_matrix)
        except np.linalg.LinAlgError as exc:
            raise RuntimeError(
                f"Could not invert (I - A); matrix may be singular. ({exc})"
            ) from exc
        self._n = n

    def solve(self, demand: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Apply the Leontief inverse to the demand vector."""
        output = self._L @ demand
        prices = np.ones(self._n)
        return output, prices
