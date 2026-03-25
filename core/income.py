"""
Income and tax calculation utilities
======================================

Functions to build VA component matrices and apply income/corporate taxes.
Extracted from the original simulation.py.
"""

from __future__ import annotations

import logging
from typing import Dict

import numpy as np

logger = logging.getLogger(__name__)


def build_va_component_matrix(isic_map: Dict[str, int]) -> Dict[str, np.ndarray]:
    """
    Build per-sector coefficient vectors for each VA component.

    Reads the cheapest production method per sector from the database
    and normalises each value-added component to
    ``$ of component per $ of output``.

    Parameters
    ----------
    isic_map : dict
        Mapping from ISIC code → sector index.

    Returns
    -------
    dict
        Keys: ``minWages``, ``bonusWages``, ``wages``, ``surplus``.
        Values: np.ndarray of shape ``(n,)``  ($ per $ of output).
    """
    from db.repositories.production_repo import ProductionsDatabase
    from db.repositories.good_repo import GoodsDatabase

    ptdb = ProductionsDatabase()
    gdb = GoodsDatabase()
    goods = gdb.get_all_goods()

    n = len(isic_map)
    minWages_coeff  = np.zeros(n)
    bonusWages_coeff = np.zeros(n)
    wages_coeff     = np.zeros(n)
    surplus_coeff   = np.zeros(n)

    for isic, idx in isic_map.items():
        good = next((g for g in goods if g.isic == isic), None)
        if not good:
            continue

        prods = ptdb.get_all_productions_by_good(int(good.id_number))
        if not prods:
            continue

        production = min(
            prods,
            key=lambda p: float(p.price) if p.price else float("inf"),
        )

        output_price = float(production.price) if production.price else 1.0
        if output_price <= 0:
            output_price = 1.0

        va = production.production_added_values
        if va:
            minWages_coeff[idx]  = float(va.get("minWages",  0)) / output_price
            bonusWages_coeff[idx] = float(va.get("bonusWages", 0)) / output_price
            wages_coeff[idx]     = float(va.get("wages",     0)) / output_price
            surplus_coeff[idx]   = float(va.get("surplus",   0)) / output_price

    return {
        "minWages":   minWages_coeff,
        "bonusWages": bonusWages_coeff,
        "wages":      wages_coeff,
        "surplus":    surplus_coeff,
    }


def apply_taxes(
    income: Dict[str, np.ndarray],
    income_tax_rate: float,
    corporate_tax_rate: float,
    income_tax_applies_to: str,
) -> Dict[str, np.ndarray]:
    """
    Apply income and corporate taxes to the income components.

    Parameters
    ----------
    income : dict
        Keys: ``minWages``, ``bonusWages``, ``wages``, ``surplus``
        (each an np.ndarray of gross values by sector).
    income_tax_rate : float
        Tax rate on wages (0.0 – 1.0).
    corporate_tax_rate : float
        Tax rate on surplus (0.0 – 1.0).
    income_tax_applies_to : str
        Which wage component to tax:
        - ``"bonusWages"`` — only bonus wages
        - ``"wages"``      — total wages
        - ``"both"``       — same as ``"wages"``

    Returns
    -------
    dict
        Gross, net, and tax arrays for each component.
        Keys: ``{component}_gross``, ``{component}_net``,
        ``income_tax``, ``corporate_tax``, ``total_tax``.
    """
    minWages  = income["minWages"].copy()
    bonusWages = income["bonusWages"].copy()
    wages     = income["wages"].copy()
    surplus   = income["surplus"].copy()

    if income_tax_applies_to == "bonusWages":
        income_tax    = bonusWages * income_tax_rate
        bonusWages_net = bonusWages * (1 - income_tax_rate)
        minWages_net  = minWages
        wages_net     = wages - income_tax
    elif income_tax_applies_to in ("wages", "both"):
        income_tax    = wages * income_tax_rate
        wages_net     = wages * (1 - income_tax_rate)
        minWages_net  = minWages * (1 - income_tax_rate)
        bonusWages_net = bonusWages * (1 - income_tax_rate)
    else:
        raise ValueError(
            f"Invalid income_tax_applies_to: '{income_tax_applies_to}'. "
            "Use 'bonusWages', 'wages', or 'both'."
        )

    corporate_tax = surplus * corporate_tax_rate
    surplus_net   = surplus * (1 - corporate_tax_rate)
    total_tax     = income_tax + corporate_tax

    return {
        "minWages_gross":  minWages,
        "bonusWages_gross": bonusWages,
        "wages_gross":     wages,
        "surplus_gross":   surplus,
        "minWages_net":    minWages_net,
        "bonusWages_net":  bonusWages_net,
        "wages_net":       wages_net,
        "surplus_net":     surplus_net,
        "income_tax":      income_tax,
        "corporate_tax":   corporate_tax,
        "total_tax":       total_tax,
    }
