"""
BaseSolver abstract base class
===============================

All solver implementations inherit from this class.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Tuple

import numpy as np


class BaseSolver(ABC):
    """
    Abstract base class for I-O model solvers.

    Sub-classes must implement :meth:`solve`, which maps a final-demand
    vector to a gross-output vector and a price vector.
    """

    @abstractmethod
    def solve(self, demand: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Solve the I-O system for a given final-demand vector.

        Parameters
        ----------
        demand : np.ndarray, shape (n,)
            Final-demand vector in money units.

        Returns
        -------
        output : np.ndarray, shape (n,)
            Gross output by sector.
        prices : np.ndarray, shape (n,)
            Market prices by sector (1.0 for Leontief; supply-curve prices
            for the dynamic equilibrium solver).
        """


