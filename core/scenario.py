"""
Scenario runner
================

Encapsulates the per-scenario iteration loop that was the private
``_run_scenario`` function in the original simulation.py.

``ScenarioRunner.run()`` executes *iterations* rounds of the circular-flow
model and returns a list of per-iteration result dicts.  The display logic
(``_display_scenario`` in the original) stays inside this class as
``display()``, keeping all scenario-level concerns together.
"""

from __future__ import annotations

import logging
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np

from core.demand import distribute_demand
from core.income import apply_taxes

logger = logging.getLogger(__name__)


class ScenarioRunner:
    """
    Runs one I-O scenario for a given number of circular-flow iterations.

    Parameters
    ----------
    name : str
        Human-readable label for this scenario (e.g. ``"Baseline Technology"``).
    solver_fn : callable
        A function ``(demand) -> (output, prices)`` – any :class:`BaseSolver`
        adapted to a plain callable.
    A_matrix : np.ndarray
        Technical coefficient matrix for the scenario.
    VA_coeffs : np.ndarray
        Value-added coefficient vector.
    va_components : dict
        Per-sector component coefficients from
        :func:`~core.income.build_va_component_matrix`.
    scale_factors : np.ndarray
        Scaling vector that maps DB VA proportions to the VA vector.
    income_tax_rate : float
    corporate_tax_rate : float
    income_tax_applies_to : str
    iterations : int
    consumption_proportions : list or None
    investment_proportions : list or None
    government_proportions : list or None
    demand_distribution : str
    initial_demand_proportions : np.ndarray or None
    domestic_indices : list of int
    n : int
    has_taxes : bool
    """

    def __init__(
        self,
        name: str,
        solver_fn: Callable[[np.ndarray], Tuple[np.ndarray, np.ndarray]],
        A_matrix: np.ndarray,
        VA_coeffs: np.ndarray,
        va_components: Dict[str, np.ndarray],
        scale_factors: np.ndarray,
        income_tax_rate: float,
        corporate_tax_rate: float,
        income_tax_applies_to: str,
        iterations: int,
        consumption_proportions: Optional[List[float]],
        investment_proportions: Optional[List[float]],
        government_proportions: Optional[List[float]],
        demand_distribution: str,
        initial_demand_proportions: Optional[np.ndarray],
        domestic_indices: List[int],
        n: int,
        has_taxes: bool,
    ):
        self.name = name
        self.solver_fn = solver_fn
        self.A_matrix = A_matrix
        self.VA_coeffs = VA_coeffs
        self.va_components = va_components
        self.scale_factors = scale_factors
        self.income_tax_rate = income_tax_rate
        self.corporate_tax_rate = corporate_tax_rate
        self.income_tax_applies_to = income_tax_applies_to
        self.iterations = iterations
        self.consumption_proportions = consumption_proportions
        self.investment_proportions = investment_proportions
        self.government_proportions = government_proportions
        self.demand_distribution = demand_distribution
        self.initial_demand_proportions = initial_demand_proportions
        self.domestic_indices = domestic_indices
        self.n = n
        self.has_taxes = has_taxes

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def run(
        self,
        demand: np.ndarray,
        verbose: bool = True,
        capital_phase: Optional[Dict[str, Any]] = None,
    ) -> List[Dict[str, Any]]:
        """
        Execute the iteration loop.

        Parameters
        ----------
        demand : np.ndarray
            Initial final-demand vector.
        verbose : bool
            Whether to print per-iteration progress.
        capital_phase : dict or None
            If supplied, enables 2-phase capital-investment simulation.
            Expected keys: ``capital_demand``, ``transition_after``,
            ``solver_fn_new``, ``A_matrix_new``, ``VA_coeffs_new``,
            ``scale_factors_new``.

        Returns
        -------
        list of dict
            One entry per iteration with keys: ``iteration``, ``phase``,
            ``demand``, ``X``, ``VA_by_sector``, ``VA_total``,
            ``intermediate_by_sector``, ``intermediate_total``,
            ``income``, ``after_tax``, ``C``, ``I``, ``G``,
            ``demand_C``, ``demand_I``, ``demand_G``,
            ``va_output_ratio``, ``intermediate_output_ratio``.
        """
        history: List[Dict] = []
        current_demand = demand.copy()
        current_C = current_I = current_G = None

        for iteration in range(self.iterations):
            # Determine active technology (capital investment phase logic)
            solver_fn, A_matrix, VA_coeffs, scale_factors, phase_name = (
                self._resolve_active_state(
                    iteration, capital_phase, current_demand
                )
            )
            if capital_phase and phase_name == "Investment":
                # Re-read the potentially modified current_demand
                current_demand = self._apply_capital_demand(
                    current_demand, capital_phase["capital_demand"]
                )

            if self.iterations > 1 and verbose:
                phase_str = f" [{phase_name}]" if phase_name else ""
                print(f"\n>>> Iteration {iteration + 1}/{self.iterations}{phase_str}")
                print(f"    Final Demand: ${current_demand.sum():>12,.2f}")

            output, prices = solver_fn(current_demand)

            VA_by_sector = VA_coeffs * output
            VA_total = VA_by_sector.sum()

            intermediate_by_sector = A_matrix.sum(axis=0) * output
            intermediate_total = intermediate_by_sector.sum()

            income = {
                "minWages":  output * self.va_components["minWages"]  * scale_factors,
                "bonusWages": output * self.va_components["bonusWages"] * scale_factors,
                "wages":     output * self.va_components["wages"]     * scale_factors,
                "surplus":   output * self.va_components["surplus"]   * scale_factors,
            }

            after_tax = apply_taxes(
                income,
                self.income_tax_rate,
                self.corporate_tax_rate,
                self.income_tax_applies_to,
            )

            C     = after_tax["wages_net"].sum()
            I_val = after_tax["surplus_net"].sum()
            G     = after_tax["total_tax"].sum()

            va_output_ratio = np.divide(
                VA_by_sector, output, where=output > 0.01,
                out=np.zeros_like(VA_by_sector)
            )
            intermediate_output_ratio = np.divide(
                intermediate_by_sector, output, where=output > 0.01,
                out=np.zeros_like(intermediate_by_sector)
            )

            history.append({
                "iteration": iteration + 1,
                "phase": phase_name,
                "demand": current_demand.copy(),
                "X": output.copy(),
                "VA_by_sector": VA_by_sector.copy(),
                "VA_total": VA_total,
                "intermediate_by_sector": intermediate_by_sector.copy(),
                "intermediate_total": intermediate_total,
                "income": income,
                "after_tax": after_tax,
                "C": C,
                "I": I_val,
                "G": G,
                "demand_C": current_C.copy() if current_C is not None else None,
                "demand_I": current_I.copy() if current_I is not None else None,
                "demand_G": current_G.copy() if current_G is not None else None,
                "va_output_ratio": va_output_ratio.copy(),
                "intermediate_output_ratio": intermediate_output_ratio.copy(),
            })

            if self.iterations > 1 and verbose:
                print(f"    Gross Output:   ${output.sum():>12,.2f}")
                print(f"    Value Added:    ${VA_total:>12,.2f}")
                print(f"      C (Consumption): ${C:>12,.2f}")
                print(f"      I (Investment):  ${I_val:>12,.2f}")
                if self.has_taxes:
                    print(f"      G (Government):  ${G:>12,.2f}")
                    print(f"      Tax Revenue:     ${after_tax['total_tax'].sum():>12,.2f}")

            # Prepare next period's demand
            if iteration < self.iterations - 1:
                current_C = distribute_demand(
                    C, self.consumption_proportions, self.demand_distribution,
                    self.initial_demand_proportions, self.domestic_indices, self.n
                )
                current_I = distribute_demand(
                    I_val, self.investment_proportions, self.demand_distribution,
                    self.initial_demand_proportions, self.domestic_indices, self.n
                )
                current_G = distribute_demand(
                    G, self.government_proportions, self.demand_distribution,
                    self.initial_demand_proportions, self.domestic_indices, self.n
                )
                current_demand = current_C + current_I + current_G

        return history

    def display(self, history: List[Dict], isic_map: Dict[str, int]) -> None:
        """
        Print the full results for this scenario (mirrors ``_display_scenario``).

        Parameters
        ----------
        history : list of dict
            Result of :meth:`run`.
        isic_map : dict
            ISIC → index mapping.
        """
        last = history[-1]
        X = last["X"]
        VA_by_sector = last["VA_by_sector"]
        VA_total = last["VA_total"]
        intermediate_by_sector = last["intermediate_by_sector"]
        intermediate_total = last["intermediate_total"]
        income = last["income"]
        at = last["after_tax"]

        if self.iterations > 1:
            print(f"\n--- Summary After {self.iterations} Iterations ---")

        print(f"\nAggregate Results:")
        print(f"  Gross Output (X):           {X.sum():>12,.2f}")
        print(f"  Final Demand (FD):          {last['demand'].sum():>12,.2f}")
        print(f"  Value Added (VA):           {VA_total:>12,.2f}")
        print(f"  Total Intermediate Inputs:  {intermediate_total:>12,.2f}")

        print(f"\n  Value Added Components (Income Distribution):")
        print(f"    Min Wages:     ${income['minWages'].sum():>12,.2f}")
        print(f"    Bonus Wages:   ${income['bonusWages'].sum():>12,.2f}")
        print(f"    Total Wages:   ${income['wages'].sum():>12,.2f}")
        print(f"    Surplus:       ${income['surplus'].sum():>12,.2f}")
        print(f"    TOTAL VA:      ${(income['wages'].sum() + income['surplus'].sum()):>12,.2f}")

        if self.has_taxes:
            print(f"\n  After-Tax Income:")
            print(
                f"    Wages (gross):   ${at['wages_gross'].sum():>12,.2f}  ->  "
                f"Wages (net):   ${at['wages_net'].sum():>12,.2f}  "
                f"(Income Tax: ${at['income_tax'].sum():>12,.2f})"
            )
            print(
                f"    Surplus (gross): ${at['surplus_gross'].sum():>12,.2f}  ->  "
                f"Surplus (net): ${at['surplus_net'].sum():>12,.2f}  "
                f"(Corp Tax:   ${at['corporate_tax'].sum():>12,.2f})"
            )
            print(f"    Total Tax Revenue: ${at['total_tax'].sum():>12,.2f}")

        print(f"\n  GDP Expenditure Components:")
        if self.has_taxes:
            print(f"    C (Consumption):  ${last['C']:>12,.2f}  (= Wages after income tax)")
            print(f"    I (Investment):   ${last['I']:>12,.2f}  (= Surplus after corporate tax)")
            print(f"    G (Government):   ${last['G']:>12,.2f}  (= Tax revenue)")
        else:
            print(f"    C (Consumption = Wages):   ${last['C']:>12,.2f}")
            print(f"    I (Investment = Surplus):  ${last['I']:>12,.2f}")
            print(f"    G (Government = Taxes):    ${last['G']:>12,.2f}")

        print(f"    FD (C + I + G):            ${(last['C'] + last['I'] + last['G']):>12,.2f}")

        self._display_sector_tables(last, X, VA_by_sector, VA_total,
                                    intermediate_by_sector, intermediate_total,
                                    income, isic_map)

        if self.iterations > 1:
            self._display_iteration_evolution(history)

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _apply_capital_demand(
        current_demand: np.ndarray, capital_demand: np.ndarray
    ) -> np.ndarray:
        """Redirect a fraction of FD toward capital-producing sectors."""
        cap = capital_demand
        total_cap = cap.sum()
        total_fd = current_demand.sum()
        if total_fd > 0 and total_cap < total_fd:
            return current_demand * (1.0 - total_cap / total_fd) + cap
        return current_demand

    def _resolve_active_state(self, iteration, capital_phase, current_demand):
        """Return (solver_fn, A, VA, scales, phase_name) for this iteration."""
        if capital_phase is None:
            return (
                self.solver_fn, self.A_matrix, self.VA_coeffs,
                self.scale_factors, None
            )

        if iteration < capital_phase["transition_after"]:
            return (
                self.solver_fn, self.A_matrix, self.VA_coeffs,
                self.scale_factors, "Investment"
            )
        return (
            capital_phase["solver_fn_new"],
            capital_phase["A_matrix_new"],
            capital_phase["VA_coeffs_new"],
            capital_phase["scale_factors_new"],
            "New Technology",
        )

    def _display_sector_tables(
        self, last, X, VA_by_sector, VA_total,
        intermediate_by_sector, intermediate_total, income, isic_map
    ):
        fd_total = last["demand"].sum()

        print(f"\nFinal Demand Composition:")
        print(f"{'Sector':<20} {'Amount':<15} {'% of Total FD':<15}")
        print("-" * 50)
        for isic, idx in sorted(isic_map.items(), key=lambda x: x[1]):
            if last["demand"][idx] > 0.01:
                fd_pct = last["demand"][idx] / fd_total * 100 if fd_total > 0 else 0
                print(f"[{idx}] {isic[:17]:<17} {last['demand'][idx]:>12.2f}   {fd_pct:>12.2f}%")
        print("-" * 50)
        print(f"{'TOTAL':<20} {fd_total:>12.2f}   {'100.00%':>15}")

        print(f"\nValue Added Composition:")
        print(f"{'Sector':<20} {'Amount':<15} {'% of Total VA':<15}")
        print("-" * 50)
        for isic, idx in sorted(isic_map.items(), key=lambda x: x[1]):
            if VA_by_sector[idx] > 0.01:
                va_pct = VA_by_sector[idx] / VA_total * 100 if VA_total > 0 else 0
                print(f"[{idx}] {isic[:17]:<17} {VA_by_sector[idx]:>12.2f}   {va_pct:>12.2f}%")
        print("-" * 50)
        print(f"{'TOTAL':<20} {VA_total:>12.2f}   {'100.00%':>15}")

        print(f"\nSector-by-Sector Breakdown:")
        print(f"{'Sector':<20} {'Output (X)':<15} {'Value Added':<15} {'Intermediate':<15}")
        print("-" * 65)
        for isic, idx in sorted(isic_map.items(), key=lambda x: x[1]):
            print(
                f"[{idx}] {isic[:17]:<17} "
                f"{X[idx]:>12.2f}   "
                f"{VA_by_sector[idx]:>12.2f}   "
                f"{intermediate_by_sector[idx]:>12.2f}"
            )
        print("-" * 65)
        print(
            f"{'TOTAL':<20} "
            f"{X.sum():>12.2f}   "
            f"{VA_total:>12.2f}   "
            f"{intermediate_total:>12.2f}"
        )

    def _display_iteration_evolution(self, history: List[Dict]):
        print(f"\n--- GDP Expenditure Evolution ({self.name}) ---")
        print(f"{'Iter':<6} {'C (Wages)':<18} {'I (Surplus)':<18} {'G (Taxes)':<18} {'FD (C+I+G)':<18}")
        print("-" * 80)
        for hist in history:
            print(
                f"{hist['iteration']:<6} "
                f"${hist['C']:>15,.2f}  "
                f"${hist['I']:>15,.2f}  "
                f"${hist['G']:>15,.2f}  "
                f"${(hist['C'] + hist['I'] + hist['G']):>15,.2f}"
            )

        print(f"\n--- Income Component Evolution ({self.name}) ---")
        print(
            f"{'Iter':<6} {'minWages':<15} {'bonusWages':<15} "
            f"{'Wages':<15} {'Surplus':<15} {'Total VA':<15}"
        )
        print("-" * 85)
        for hist in history:
            inc = hist["income"]
            print(
                f"{hist['iteration']:<6} "
                f"${inc['minWages'].sum():>12,.2f}  "
                f"${inc['bonusWages'].sum():>12,.2f}  "
                f"${inc['wages'].sum():>12,.2f}  "
                f"${inc['surplus'].sum():>12,.2f}  "
                f"${hist['VA_total']:>12,.2f}"
            )
