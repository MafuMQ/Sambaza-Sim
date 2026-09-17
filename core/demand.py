"""
Demand vector builders
=======================

Pure functions to construct a final-demand vector from one of several
specification modes. Extracted from the original simulation.py.
"""

from __future__ import annotations

import random
import numpy as np
from typing import Dict, List, Optional


def build_demand_vector(
    n_sectors: int,
    isic_map: Dict[str, int],
    final_demand: Optional[np.ndarray] = None,
    demand_vector: Optional[np.ndarray] = None,
    target_isic: Optional[str] = None,
    demand_shock: Optional[float] = None,
    uniform_demand: Optional[float] = None,
    total_demand: Optional[float] = None,
    proportions: Optional[List[float]] = None,
) -> np.ndarray:
    """
    Build a demand vector from one of several specification modes.

    Modes (evaluated in order of precedence)
    -----------------------------------------
    1. ``final_demand``  — direct vector
    2. ``demand_vector`` — direct vector (alias for final_demand)
    3. ``target_isic`` + ``demand_shock`` — single sector
    4. ``demand_shock`` alone — random sector
    5. ``uniform_demand`` — same amount for all domestic sectors
    6. ``total_demand`` + ``proportions`` — proportional distribution

    Parameters
    ----------
    n_sectors : int
        Number of sectors (length of the vector to build).
    isic_map : dict
        Mapping from ISIC code → column/row index.

    Returns
    -------
    np.ndarray
        Demand vector of shape ``(n_sectors,)``.

    Raises
    ------
    ValueError
        If none of the specification modes can be satisfied from the
        provided arguments.
    """
    if final_demand is not None:
        demand = np.array(final_demand, dtype=float)
        if len(demand) != n_sectors:
            raise ValueError(
                f"final_demand length {len(demand)} != {n_sectors} sectors"
            )
        print("\nDemand Mode: Direct final demand vector")
        return demand

    if demand_vector is not None:
        demand = np.array(demand_vector, dtype=float)
        if len(demand) != n_sectors:
            raise ValueError(
                f"demand_vector length {len(demand)} != {n_sectors} sectors"
            )
        print("\nDemand Mode: Direct demand vector")
        return demand

    if target_isic is not None and demand_shock is not None:
        if target_isic not in isic_map:
            raise ValueError(f"Target ISIC '{target_isic}' not found in sector map")
        demand = np.zeros(n_sectors)
        idx = isic_map[target_isic]
        demand[idx] = demand_shock
        print("\nDemand Mode: Sector-specific")
        print(f"  Target: [{idx}] {target_isic}")
        print(f"  Demand: ${demand_shock:,.2f}")
        return demand

    if demand_shock is not None:
        domestic_isics = [k for k in isic_map if k != "A9999_999_999"]
        if not domestic_isics:
            raise ValueError("No domestic sectors found")
        chosen_isic = random.choice(domestic_isics)
        demand = np.zeros(n_sectors)
        idx = isic_map[chosen_isic]
        demand[idx] = demand_shock
        print("\nDemand Mode: Random sector")
        print(f"  Selected: [{idx}] {chosen_isic}")
        print(f"  Demand: ${demand_shock:,.2f}")
        return demand

    if uniform_demand is not None:
        demand = np.zeros(n_sectors, dtype=float)
        for isic, idx in isic_map.items():
            if isic != "A9999_999_999":
                demand[idx] = uniform_demand
        domestic_count = int((demand > 0).sum())
        print(
            f"\nDemand Mode: Uniform (${uniform_demand:,.2f} per sector, "
            f"{domestic_count} domestic sectors)"
        )
        return demand

    if total_demand is not None and proportions is not None:
        props = np.array(proportions, dtype=float)
        if len(props) != n_sectors:
            raise ValueError(
                f"Proportions length {len(props)} != {n_sectors} sectors"
            )
        if not np.isclose(props.sum(), 1.0):
            raise ValueError(f"Proportions must sum to 1.0, got {props.sum()}")
        demand = props * total_demand
        print("\nDemand Mode: Proportional distribution")
        print(f"  Total demand: ${total_demand:,.2f}")
        return demand

    raise ValueError(
        "Must provide one of: final_demand, demand_vector, "
        "(target_isic + demand_shock), demand_shock, uniform_demand, "
        "or (total_demand + proportions)"
    )


def distribute_demand(
    amount: float,
    explicit_proportions: Optional[List[float]],
    demand_distribution: str,
    initial_demand_proportions: Optional[np.ndarray],
    domestic_indices: List[int],
    n: int,
) -> np.ndarray:
    """
    Distribute an aggregate demand amount across sectors.

    Order of precedence
    -------------------
    1. *explicit_proportions* (if given) — must sum to 1.0
    2. *demand_distribution* == ``"proportional"`` with *initial_demand_proportions*
    3. *demand_distribution* == ``"uniform"`` across domestic sectors
    4. Fallback: *initial_demand_proportions* or uniform across all sectors

    Parameters
    ----------
    amount : float
        The aggregate dollar amount to distribute.
    explicit_proportions : list or None
        Explicit weight for each sector (length == n, sums to 1.0).
    demand_distribution : str
        Fallback distribution strategy: ``"proportional"`` or ``"uniform"``.
    initial_demand_proportions : np.ndarray or None
        Base proportions derived from the initial demand vector.
    domestic_indices : list of int
        Indices of domestic (non-import) sectors.
    n : int
        Total number of sectors.

    Returns
    -------
    np.ndarray  of shape ``(n,)``
    """
    if amount == 0:
        return np.zeros(n)

    if explicit_proportions is not None:
        props = np.array(explicit_proportions, dtype=float)
        if len(props) != n:
            raise ValueError(
                f"Proportions length {len(props)} does not match sectors {n}"
            )
        if not np.isclose(props.sum(), 1.0, atol=1e-2):
            raise ValueError(f"Proportions must sum to approximately 1.0, got {props.sum()}")
        # Normalize to ensure exactly 1.0
        props = props / props.sum()
        return props * amount

    if demand_distribution == "proportional" and initial_demand_proportions is not None:
        return initial_demand_proportions * amount

    if demand_distribution == "uniform":
        result = np.zeros(n)
        per_sector = amount / len(domestic_indices) if domestic_indices else 0
        for i in domestic_indices:
            result[i] = per_sector
        return result

    # Fallback
    if initial_demand_proportions is not None:
        return initial_demand_proportions * amount
    return np.full(n, amount / n)


