"""
Simulation orchestrator
========================

The ``Simulation`` class ties all core components together and exposes a
single ``run()`` method.  The module-level ``run_simulation()`` function
remains the public interface used by the UI and demos; it creates a
``Simulation`` object internally for backward compatibility.

Key identity (maintained throughout): VA = FD = C + I + G
  C = Wages net of income tax
  I = Surplus net of corporate tax
  G = Total tax revenue
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

import numpy as np

from core.demand import build_demand_vector
from core.income import apply_taxes, build_va_component_matrix
from core.scenario import ScenarioRunner
from core.solvers.leontief import LeontiefSolver
from core.solvers.supply_curves import (
    DynamicEquilibriumSolver,
    SupplyCurveSolver,
    load_supply_curves_from_db,
)
from core.io_matrix import build_io_matrix

logger = logging.getLogger(__name__)


class Simulation:
    """
    Orchestrates a before/after scenario comparison.

    Construct with keyword arguments matching the ``run_simulation()``
    parameter list (see that function's docstring for full documentation).

    Call :meth:`run` to execute and get results.
    """

    def __init__(self, **kwargs):
        self._params = kwargs

    def run(self) -> Optional[Dict[str, Any]]:
        """Execute the simulation and return the results dict."""
        return _run_simulation_impl(**self._params)


# ---------------------------------------------------------------------------
# Public entry-point (backward compatible)
# ---------------------------------------------------------------------------


def run_simulation(
    # ── Demand specification (provide exactly ONE) ─────────────────────────
    final_demand=None,
    demand_vector=None,
    target_isic=None,
    demand_shock=None,
    uniform_demand=None,
    total_demand=None,
    proportions=None,
    # ── Technology ─────────────────────────────────────────────────────────
    A_before=None,
    VA_before=None,
    A_after=None,
    VA_after=None,
    isic_map=None,
    tech_change=None,
    # ── Tax policy ─────────────────────────────────────────────────────────
    income_tax_rate_before: float = 0.0,
    income_tax_rate_after: float = 0.0,
    corporate_tax_rate_before: float = 0.0,
    corporate_tax_rate_after: float = 0.0,
    income_tax_applies_to: str = "bonusWages",
    # ── Circular flow ──────────────────────────────────────────────────────
    consumption_proportions=None,
    investment_proportions=None,
    government_proportions=None,
    iterations: int = 1,
    demand_distribution: str = "proportional",
    consumption_rate: float = 1.0,
    # ── Solver ─────────────────────────────────────────────────────────────
    solver_type: str = "leontief",
    # ── Display ────────────────────────────────────────────────────────────
    before_name: str = "Before",
    after_name: str = "After",
    loggingLevel: int = logging.WARNING,
) -> Optional[Dict[str, Any]]:
    """
    Run a full before/after I-O simulation.

    This is the primary public API maintained for backward compatibility
    with the UI and demo scripts.  Internally it delegates to a
    ``Simulation`` object.

    Parameters
    ----------
    (See implementation plan / original simulation.py docstring for the
    full list – the signature mirrors the original ``run_simulation()``.)

    Returns
    -------
    dict or None
        Keys: ``before``, ``after``, ``deltas``, ``config``,
        ``before_history``, ``after_history``, ``isic_map``.
    """
    return _run_simulation_impl(
        final_demand=final_demand,
        demand_vector=demand_vector,
        target_isic=target_isic,
        demand_shock=demand_shock,
        uniform_demand=uniform_demand,
        total_demand=total_demand,
        proportions=proportions,
        A_before=A_before,
        VA_before=VA_before,
        A_after=A_after,
        VA_after=VA_after,
        isic_map=isic_map,
        tech_change=tech_change,
        income_tax_rate_before=income_tax_rate_before,
        income_tax_rate_after=income_tax_rate_after,
        corporate_tax_rate_before=corporate_tax_rate_before,
        corporate_tax_rate_after=corporate_tax_rate_after,
        income_tax_applies_to=income_tax_applies_to,
        consumption_proportions=consumption_proportions,
        investment_proportions=investment_proportions,
        government_proportions=government_proportions,
        iterations=iterations,
        demand_distribution=demand_distribution,
        consumption_rate=consumption_rate,
        solver_type=solver_type,
        before_name=before_name,
        after_name=after_name,
        loggingLevel=loggingLevel,
    )


def run_tax_policy_simulation(**kwargs) -> Optional[Dict[str, Any]]:
    """
    Backward-compatible alias for :func:`run_simulation`.

    Maps old parameter names and applies supply-curve solver as default.
    """
    kwargs.setdefault("solver_type", "supply_curves")
    kwargs.setdefault("before_name", "Before Tax Change")
    kwargs.setdefault("after_name", "After Tax Change")
    if "consumption_distribution" in kwargs:
        kwargs.setdefault("demand_distribution", kwargs.pop("consumption_distribution"))
    return run_simulation(**kwargs)


# ---------------------------------------------------------------------------
# Core implementation
# ---------------------------------------------------------------------------


def _run_simulation_impl(
    final_demand=None,
    demand_vector=None,
    target_isic=None,
    demand_shock=None,
    uniform_demand=None,
    total_demand=None,
    proportions=None,
    A_before=None,
    VA_before=None,
    A_after=None,
    VA_after=None,
    isic_map=None,
    tech_change=None,
    income_tax_rate_before=0.0,
    income_tax_rate_after=0.0,
    corporate_tax_rate_before=0.0,
    corporate_tax_rate_after=0.0,
    income_tax_applies_to="bonusWages",
    consumption_proportions=None,
    investment_proportions=None,
    government_proportions=None,
    iterations=1,
    demand_distribution="proportional",
    consumption_rate=1.0,
    solver_type="leontief",
    before_name="Before",
    after_name="After",
    loggingLevel=logging.WARNING,
) -> Optional[Dict[str, Any]]:
    """Internal implementation — called by both Simulation.run() and run_simulation()."""

    # ------------------------------------------------------------------
    # 1. Build I-O matrices (if not provided)
    # ------------------------------------------------------------------
    if A_before is None:
        print("=" * 100)
        print("BUILDING COEFFICIENT MATRICES FROM DATABASE")
        print("=" * 100)
        A_before, VA_before, isic_map = build_io_matrix(
            demoDB=False, loggingLevel=loggingLevel
        )
        print(f"\n[OK] IO Matrix Built. Sectors: {len(isic_map)}")

    # ------------------------------------------------------------------
    # 1b. Apply multi-level technological change (if provided)
    # ------------------------------------------------------------------
    supply_data_after = None

    if tech_change is not None:
        from core.tech_change import LEVEL_MATRIX, LEVEL_PRODUCTION, LEVEL_CURVE
        from db.repositories.production_repo import ProductionsDatabase
        from db.repositories.supply_curve_repo import SupplyCurveDatabase

        change_levels = tech_change.get_change_levels()
        print("\n" + "=" * 100)
        print(f"APPLYING MULTI-LEVEL TECHNOLOGICAL CHANGE: {tech_change.name}")
        print("=" * 100)
        print(f"Change levels: {', '.join(change_levels)}")

        if A_after is None:
            A_after = A_before.copy()
        if VA_after is None:
            VA_after = VA_before.copy()

        if tech_change.has_production_changes():
            print(f"\n[Level 3: Productions] Applying {len(tech_change.production_changes)} changes…")
            ptdb = ProductionsDatabase()
            scdb = SupplyCurveDatabase()
            result = tech_change.apply_to_productions(
                ptdb, scdb, rebuild_curves=True, rebuild_matrix=True,
                loggingLevel=loggingLevel,
            )
            if result["A_matrix"] is not None:
                A_after = result["A_matrix"]
                VA_after = result["VA_vector"]
                isic_map = result["isic_map"]
                print(
                    f"  [OK] Cascade rebuild: {len(result['productions'])} "
                    "productions → curves → coefficient matrix"
                )
            if result.get("supply_data"):
                supply_data_after = result["supply_data"]

        if tech_change.has_curve_changes():
            print(f"\n[Level 2: Curves] Applying {len(tech_change.curve_changes)} changes…")
            ptdb = ProductionsDatabase()
            scdb = SupplyCurveDatabase()
            result = tech_change.apply_to_curves(
                scdb, ptdb=ptdb, isic_map=isic_map, rebuild_matrix=True,
                loggingLevel=loggingLevel,
            )
            supply_data_after = result["supply_data"]
            if result["A_matrix"] is not None:
                A_after = result["A_matrix"]
                VA_after = result["VA_vector"]
                isic_map = result["isic_map"]
            print(
                f"  [OK] Cascade rebuild: {len(result['modified_curves'])} "
                "curves → coefficient matrix"
            )

        if tech_change.has_matrix_changes():
            print(f"\n[Matrix Level] Applying {len(tech_change.matrix_changes)} changes…")
            A_after, VA_after = tech_change.apply(A_after, VA_after, isic_map)
            print("  [OK] A matrix and VA coefficients updated")

        print(tech_change.get_summary())

    elif A_after is None:
        A_after = A_before.copy()

    if VA_after is None:
        VA_after = VA_before.copy()

    n = A_before.shape[0]

    # ------------------------------------------------------------------
    # 2. Determine what's changing
    # ------------------------------------------------------------------
    has_tech_change = not (
        np.allclose(A_before, A_after) and np.allclose(VA_before, VA_after)
    )
    has_tax_change  = (
        income_tax_rate_before != income_tax_rate_after
        or corporate_tax_rate_before != corporate_tax_rate_after
    )
    has_taxes_before = income_tax_rate_before > 0 or corporate_tax_rate_before > 0
    has_taxes_after  = income_tax_rate_after  > 0 or corporate_tax_rate_after  > 0
    has_any_taxes = has_taxes_before or has_taxes_after

    # ------------------------------------------------------------------
    # 3. Print header
    # ------------------------------------------------------------------
    sim_type = []
    if has_tech_change:
        sim_type.append("Technological Change")
    if has_tax_change:
        sim_type.append("Tax Policy Change")
    if not sim_type:
        sim_type.append("Scenario Comparison")

    print("\n" + "=" * 100)
    print(f"INPUT-OUTPUT SIMULATION: {' + '.join(sim_type)}")
    if iterations > 1:
        print(f"(Circular Flow Model — {iterations} iterations)")
    print("=" * 100)

    if has_any_taxes or has_tax_change:
        print(f"\nTax Policy Configuration:")
        print(f"  Income Tax ({before_name}):    {income_tax_rate_before*100:.1f}% on {income_tax_applies_to}")
        print(f"  Income Tax ({after_name}):     {income_tax_rate_after*100:.1f}% on {income_tax_applies_to}")
        print(f"  Corporate Tax ({before_name}): {corporate_tax_rate_before*100:.1f}% on surplus")
        print(f"  Corporate Tax ({after_name}):  {corporate_tax_rate_after*100:.1f}% on surplus")

    if iterations > 1:
        print(f"\nCircular Flow Parameters:")
        print(f"  Iterations: {iterations}")
        print(f"  Distribution: {demand_distribution}")
        if consumption_proportions is not None:
            print(f"  Consumption Proportions: {consumption_proportions}")
        if investment_proportions is not None:
            print(f"  Investment Proportions:   {investment_proportions}")
        if government_proportions is not None:
            print(f"  Government Proportions:   {government_proportions}")

    # ------------------------------------------------------------------
    # 4. Build demand vector
    # ------------------------------------------------------------------
    print("\n" + "=" * 100)
    print("DEMAND SPECIFICATION")
    print("=" * 100)

    demand = build_demand_vector(
        n, isic_map,
        final_demand=final_demand,
        demand_vector=demand_vector,
        target_isic=target_isic,
        demand_shock=demand_shock,
        uniform_demand=uniform_demand,
        total_demand=total_demand,
        proportions=proportions,
    )

    print(f"\nDemand Vector:")
    for isic, idx in sorted(isic_map.items(), key=lambda x: x[1]):
        if isic != "A9999_999_999":
            print(f"  [{idx}] {isic[:20]:<20}: ${demand[idx]:>10.2f}")
    print(f"\nTotal Final Demand: ${demand.sum():,.2f}")

    # ------------------------------------------------------------------
    # 5. Compute shared setup
    # ------------------------------------------------------------------
    va_components = build_va_component_matrix(isic_map)
    db_va_total = va_components["wages"] + va_components["surplus"]
    scale_factors_before = np.divide(
        VA_before, db_va_total, where=db_va_total != 0,
        out=np.ones_like(VA_before)
    )
    scale_factors_after = np.divide(
        VA_after, db_va_total, where=db_va_total != 0,
        out=np.ones_like(VA_after)
    )

    print(f"\nValue Added Component Coefficients ($ per $ of output):")
    print(f"{'Sector':<20} {'minWages':<12} {'bonusWages':<12} {'wages':<12} {'surplus':<12} {'Total VA':<12}")
    print("-" * 95)
    for isic, idx in sorted(isic_map.items(), key=lambda x: x[1]):
        print(
            f"[{idx}] {isic[:17]:<17} "
            f"{va_components['minWages'][idx]:>10.4f}  "
            f"{va_components['bonusWages'][idx]:>10.4f}  "
            f"{va_components['wages'][idx]:>10.4f}  "
            f"{va_components['surplus'][idx]:>10.4f}  "
            f"{VA_before[idx]:>10.4f}"
        )

    domestic_indices = [idx for isic, idx in isic_map.items() if isic != "A9999_999_999"]
    total_domestic = sum(demand[i] for i in domestic_indices)
    initial_demand_proportions = None
    if total_domestic > 0:
        initial_demand_proportions = np.zeros(n)
        for i in domestic_indices:
            initial_demand_proportions[i] = demand[i] / total_domestic

    # ------------------------------------------------------------------
    # 6. Create solvers
    # ------------------------------------------------------------------
    def _make_solver_fn(solver_obj) -> callable:
        def solve(d):
            result = solver_obj.solve(d, verbose=False)
            if not result or result["status"] != "converged":
                raise RuntimeError("Solver failed to converge")
            return result["output"], result["prices"]
        return solve

    if solver_type == "leontief":
        try:
            solver_before_fn = LeontiefSolver(A_before).solve
        except RuntimeError as exc:
            print(f"ERROR: Could not invert (I-A) for before scenario! {exc}")
            return None
        try:
            solver_after_fn = LeontiefSolver(A_after).solve
        except RuntimeError as exc:
            print(f"ERROR: Could not invert (I-A) for after scenario! {exc}")
            return None

    elif solver_type == "supply_curves":
        print("\n" + "=" * 100)
        print("LOADING SUPPLY CURVES")
        print("=" * 100)
        supply_data = load_supply_curves_from_db(isic_map)
        print(f"[OK] Loaded supply curves for {len(supply_data)} sectors.")

        solver_before_obj = DynamicEquilibriumSolver(A_before, isic_map, supply_data)

        if supply_data_after is not None:
            print(f"[OK] Using modified supply curves for '{after_name}' scenario.")
            solver_after_obj = DynamicEquilibriumSolver(A_after, isic_map, supply_data_after)
        elif has_tech_change:
            solver_after_obj = DynamicEquilibriumSolver(A_after, isic_map, supply_data)
        else:
            solver_after_obj = solver_before_obj

        solver_before_fn = _make_solver_fn(solver_before_obj)
        solver_after_fn  = _make_solver_fn(solver_after_obj)

    else:
        raise ValueError(
            f"Unknown solver_type: '{solver_type}'. Use 'leontief' or 'supply_curves'."
        )

    # ------------------------------------------------------------------
    # 7. Run BEFORE scenario
    # ------------------------------------------------------------------
    common_kw = dict(
        va_components=va_components,
        consumption_proportions=consumption_proportions,
        investment_proportions=investment_proportions,
        government_proportions=government_proportions,
        demand_distribution=demand_distribution,
        initial_demand_proportions=initial_demand_proportions,
        domestic_indices=domestic_indices,
        n=n,
        income_tax_applies_to=income_tax_applies_to,
        iterations=iterations,
    )

    print(f"\n{'='*100}")
    if iterations > 1:
        print(f"SCENARIO 1: {before_name} (Circular Flow — {iterations} iterations)")
    else:
        print(f"SCENARIO 1: {before_name}")
    print(f"{'='*100}")

    runner_before = ScenarioRunner(
        name=before_name,
        solver_fn=solver_before_fn,
        A_matrix=A_before,
        VA_coeffs=VA_before,
        scale_factors=scale_factors_before,
        income_tax_rate=income_tax_rate_before,
        corporate_tax_rate=corporate_tax_rate_before,
        has_taxes=has_taxes_before,
        **common_kw,
    )
    before_history = runner_before.run(demand)
    runner_before.display(before_history, isic_map)

    # ------------------------------------------------------------------
    # 8. Run AFTER scenario (with optional capital investment phase)
    # ------------------------------------------------------------------
    capital_phase = None
    if tech_change is not None and tech_change.has_capital_requirements():
        capital_demand = tech_change.get_capital_demand_vector(isic_map, n)
        transition_after = tech_change.investment_duration
        min_iters = transition_after + 1
        if iterations < min_iters:
            print(f"\n[!] Capital investment requires at least {min_iters} iterations — auto-adjusting.")
            iterations = min_iters

        capital_phase = {
            "capital_demand":   capital_demand,
            "transition_after": transition_after,
            "solver_fn_new":    solver_after_fn,
            "A_matrix_new":     A_after,
            "VA_coeffs_new":    VA_after,
            "scale_factors_new": scale_factors_after,
        }

        print(f"\n{'='*100}")
        print(f"SCENARIO 2: {after_name} (2-Phase: Investment → Technology Change)")
        print(f"{'='*100}")
        print(f"\n  Capital Investment Required:")
        for sector, amount in tech_change.capital_requirements.items():
            print(f"    {sector}: ${amount:,.2f}")
        print(f"    TOTAL: ${tech_change.get_total_capital_cost():,.2f}")
        print(f"\n  Phase 1 (Iterations 1-{transition_after}): Investment")
        print(f"  Phase 2 (Iterations {transition_after + 1}-{iterations}): New Technology")

        after_solver_fn = solver_before_fn
        after_A = A_before
        after_VA = VA_before
        after_scales = scale_factors_before
    else:
        print(f"\n{'='*100}")
        if iterations > 1:
            print(f"SCENARIO 2: {after_name} (Circular Flow — {iterations} iterations)")
        else:
            print(f"SCENARIO 2: {after_name}")
        print(f"{'='*100}")

        after_solver_fn = solver_after_fn
        after_A = A_after
        after_VA = VA_after
        after_scales = scale_factors_after

    runner_after = ScenarioRunner(
        name=after_name,
        solver_fn=after_solver_fn,
        A_matrix=after_A,
        VA_coeffs=after_VA,
        scale_factors=after_scales,
        income_tax_rate=income_tax_rate_after,
        corporate_tax_rate=corporate_tax_rate_after,
        has_taxes=has_taxes_after,
        **common_kw,
    )
    after_history = runner_after.run(demand, capital_phase=capital_phase)
    runner_after.display(after_history, isic_map)

    # ------------------------------------------------------------------
    # 9. Display comparison
    # ------------------------------------------------------------------
    deltas = _display_comparison(
        before_history, after_history,
        before_name, after_name,
        isic_map, has_any_taxes, iterations,
        demand.sum(), has_tech_change, has_tax_change,
    )

    # ------------------------------------------------------------------
    # 10. Summary
    # ------------------------------------------------------------------
    print("\n" + "=" * 100)
    print("SIMULATION COMPLETE")
    print("=" * 100)

    last_b = before_history[-1]
    last_a = after_history[-1]

    if has_tech_change:
        eff_b = last_b["VA_total"] / last_b["X"].sum() if last_b["X"].sum() > 0 else 0
        eff_a = last_a["VA_total"] / last_a["X"].sum() if last_a["X"].sum() > 0 else 0
        print(f"\nResource Efficiency (VA / Gross Output):")
        print(f"  {before_name}:  {eff_b:.4f}")
        print(f"  {after_name}:   {eff_a:.4f}")
        print(f"  Change:         {(eff_a - eff_b):+.4f}")

        delta_X_total = deltas["X"].sum()
        va_gain = deltas["VA"]

        print(f"\n{'='*100}")
        print(f"TECH CHANGE METRICS")
        print(f"{'='*100}")
        print(f"  Total output gain:    {delta_X_total:>+12,.2f}")
        pct = (last_a["X"].sum() / last_b["X"].sum() - 1) * 100 if last_b["X"].sum() > 0 else 0.0
        print(f"  Percent output gain:  {pct:>+11,.4f}%")
        print(f"  Value-added gain:     {va_gain:>+12,.2f}")

        if tech_change is not None and tech_change.has_capital_requirements():
            impl_cost = tech_change.get_total_capital_cost()
            if impl_cost > 0:
                cost_eff = delta_X_total / impl_cost
                print(f"  Implementation cost:  ${impl_cost:>11,.2f}")
                print(f"  Cost effectiveness:   {cost_eff:>+12,.4f}  (Δ output per $ invested)")
        else:
            print("  Cost effectiveness:   N/A  (no capital cost specified)")

    # ------------------------------------------------------------------
    # 11. Return structured result
    # ------------------------------------------------------------------
    # Compute Leontief inverses for display (best-effort)
    try:
        L_before = np.linalg.inv(np.eye(n) - A_before)
    except np.linalg.LinAlgError:
        L_before = None
    try:
        L_after = np.linalg.inv(np.eye(n) - A_after)
    except np.linalg.LinAlgError:
        L_after = None

    return {
        "before": {
            "name":  before_name,
            "X":     last_b["X"],
            "VA":    last_b["VA_total"],
            "VA_by_sector":         last_b["VA_by_sector"],
            "FD":    demand.sum(),
            "intermediate_inputs":  last_b["intermediate_total"],
            "intermediate_by_sector": last_b["intermediate_by_sector"],
            "income":    last_b["income"],
            "after_tax": last_b["after_tax"],
            "C": last_b["C"],
            "I": last_b["I"],
            "G": last_b["G"],
        },
        "after": {
            "name":  after_name,
            "X":     last_a["X"],
            "VA":    last_a["VA_total"],
            "VA_by_sector":         last_a["VA_by_sector"],
            "FD":    demand.sum(),
            "intermediate_inputs":  last_a["intermediate_total"],
            "intermediate_by_sector": last_a["intermediate_by_sector"],
            "income":    last_a["income"],
            "after_tax": last_a["after_tax"],
            "C": last_a["C"],
            "I": last_a["I"],
            "G": last_a["G"],
        },
        "deltas": deltas,
        "config": {
            "income_tax_rate_before":  income_tax_rate_before,
            "income_tax_rate_after":   income_tax_rate_after,
            "corporate_tax_rate_before": corporate_tax_rate_before,
            "corporate_tax_rate_after":  corporate_tax_rate_after,
            "income_tax_applies_to":   income_tax_applies_to,
            "has_tech_change": has_tech_change,
            "has_tax_change":  has_tax_change,
            "has_any_taxes":   has_any_taxes,
            "iterations":  iterations,
            "solver_type": solver_type,
        },
        "before_history": before_history,
        "after_history":  after_history,
        "isic_map": isic_map,
        # Matrices for display
        "A_before": A_before,
        "A_after":  A_after,
        "VA_before": VA_before,
        "VA_after":  VA_after,
        "L_before": L_before,
        "L_after":  L_after,
        "demand_vector": demand,
    }


# ---------------------------------------------------------------------------
# Comparison display (extracted from original _display_comparison)
# ---------------------------------------------------------------------------


def _display_comparison(
    before_history, after_history,
    before_name, after_name,
    isic_map, has_taxes, iterations,
    initial_demand_sum, has_tech_change, has_tax_change,
) -> Dict[str, Any]:
    """Print the full before/after comparison table and return a deltas dict."""
    last_b = before_history[-1]
    last_a = after_history[-1]

    X_b, X_a = last_b["X"], last_a["X"]
    VA_b, VA_a = last_b["VA_by_sector"], last_a["VA_by_sector"]
    VA_total_b, VA_total_a = last_b["VA_total"], last_a["VA_total"]
    int_b, int_a = last_b["intermediate_by_sector"], last_a["intermediate_by_sector"]
    int_total_b, int_total_a = last_b["intermediate_total"], last_a["intermediate_total"]
    inc_b, inc_a = last_b["income"], last_a["income"]
    at_b, at_a = last_b["after_tax"], last_a["after_tax"]

    delta_X = X_a - X_b
    delta_VA_by_sector = VA_a - VA_b
    delta_VA = VA_total_a - VA_total_b
    delta_int_by_sector = int_a - int_b
    delta_int = int_total_a - int_total_b

    if has_tech_change and has_tax_change:
        title = "Impact of Combined Technology & Tax Policy Change"
    elif has_tech_change:
        title = "Impact of Technological Change"
    else:
        title = "Impact of Tax Policy Change"

    print(f"\n{'='*100}")
    print(f"COMPARISON: {title}")
    print(f"{'='*100}")

    print(f"\nAggregate Changes (initial FD = ${initial_demand_sum:,.2f}):")
    print(f"  Change in Gross Output:     {delta_X.sum():>+12,.2f}  ({(delta_X.sum()/X_b.sum()*100):>+7.2f}%)")
    print(f"  Change in Value Added:      {delta_VA:>+12,.2f}  ({(delta_VA/VA_total_b*100):>+7.2f}%)")
    d_int_pct = delta_int / int_total_b * 100 if int_total_b > 0 else 0
    print(f"  Change in Interm. Inputs:   {delta_int:>+12,.2f}  ({d_int_pct:>+7.2f}%)")

    d_minw   = inc_a["minWages"].sum()  - inc_b["minWages"].sum()
    d_bonusw = inc_a["bonusWages"].sum() - inc_b["bonusWages"].sum()
    d_wages  = inc_a["wages"].sum()     - inc_b["wages"].sum()
    d_surplus = inc_a["surplus"].sum()  - inc_b["surplus"].sum()

    print(f"\n  Value Added Component Changes:")

    def _pct(num, denom):
        return num / denom * 100 if denom > 0 else 0

    print(f"    ΔMin Wages:    {d_minw:>+12,.2f}  ({_pct(d_minw, inc_b['minWages'].sum()):>+7.2f}%)")
    print(f"    ΔBonus Wages:  {d_bonusw:>+12,.2f}  ({_pct(d_bonusw, inc_b['bonusWages'].sum()):>+7.2f}%)")
    print(f"    ΔTotal Wages:  {d_wages:>+12,.2f}  ({_pct(d_wages, inc_b['wages'].sum()):>+7.2f}%)")
    print(f"    ΔSurplus:      {d_surplus:>+12,.2f}  ({_pct(d_surplus, inc_b['surplus'].sum()):>+7.2f}%)")

    d_C = last_a["C"] - last_b["C"]
    d_I = last_a["I"] - last_b["I"]
    d_G = last_a["G"] - last_b["G"]

    print(f"\n  GDP Expenditure Component Changes:")
    print(f"    ΔC (Consumption): {d_C:>+12,.2f}  ({_pct(d_C, last_b['C']):>+7.2f}%)")
    print(f"    ΔI (Investment):  {d_I:>+12,.2f}  ({_pct(d_I, last_b['I']):>+7.2f}%)")
    print(f"    ΔG (Government):  {d_G:>+12,.2f}")

    if has_taxes:
        d_income_tax  = at_a["income_tax"].sum()    - at_b["income_tax"].sum()
        d_corp_tax    = at_a["corporate_tax"].sum()  - at_b["corporate_tax"].sum()
        d_total_tax   = at_a["total_tax"].sum()      - at_b["total_tax"].sum()
        d_wages_net   = at_a["wages_net"].sum()      - at_b["wages_net"].sum()
        d_surplus_net = at_a["surplus_net"].sum()    - at_b["surplus_net"].sum()

        print(f"\n  Tax Revenue Changes:")
        print(f"    ΔIncome Tax:     {d_income_tax:>+12,.2f}  ({_pct(d_income_tax, at_b['income_tax'].sum()):>+7.2f}%)")
        print(f"    ΔCorporate Tax:  {d_corp_tax:>+12,.2f}  ({_pct(d_corp_tax, at_b['corporate_tax'].sum()):>+7.2f}%)")
        print(f"    ΔTotal Tax:      {d_total_tax:>+12,.2f}  ({_pct(d_total_tax, at_b['total_tax'].sum()):>+7.2f}%)")
        print(f"\n  After-Tax Income Changes:")
        print(f"    ΔWages (net):    {d_wages_net:>+12,.2f}  ({_pct(d_wages_net, at_b['wages_net'].sum()):>+7.2f}%)")
        print(f"    ΔSurplus (net):  {d_surplus_net:>+12,.2f}  ({_pct(d_surplus_net, at_b['surplus_net'].sum()):>+7.2f}%)")

        print(f"\n  Distribution Impact:")
        if d_wages_net > 0:
            print(f"    Workers gain ${d_wages_net:,.2f}")
        elif d_wages_net < 0:
            print(f"    Workers lose ${-d_wages_net:,.2f}")
        if d_surplus_net > 0:
            print(f"    Businesses gain ${d_surplus_net:,.2f}")
        elif d_surplus_net < 0:
            print(f"    Businesses lose ${-d_surplus_net:,.2f}")
        if d_total_tax > 0:
            print(f"    Government gains ${d_total_tax:,.2f}")
        elif d_total_tax < 0:
            print(f"    Government loses ${-d_total_tax:,.2f}")

    # Sector table
    print(f"\nDetailed Sector-by-Sector Changes:")
    print(f"{'Sector':<20} {'ΔOutput':<15} {'% Chg':<10} {'ΔValue Added':<15} {'ΔIntermediate':<15}")
    print("-" * 85)
    for isic, idx in sorted(isic_map.items(), key=lambda x: x[1]):
        pct = (X_a[idx] - X_b[idx]) / X_b[idx] * 100 if X_b[idx] != 0 else 0
        print(
            f"[{idx}] {isic[:17]:<17} "
            f"{delta_X[idx]:>+12.2f}   "
            f"{pct:>+7.2f}%   "
            f"{delta_VA_by_sector[idx]:>+12.2f}   "
            f"{delta_int_by_sector[idx]:>+12.2f}"
        )
    print("-" * 85)
    print(
        f"{'TOTAL':<20} "
        f"{delta_X.sum():>+12.2f}   "
        f"{(delta_X.sum()/X_b.sum()*100):>+7.2f}%   "
        f"{delta_VA:>+12.2f}   "
        f"{delta_int:>+12.2f}"
    )

    if iterations > 1:
        print(f"\n{'='*100}")
        print("ITERATION-BY-ITERATION COMPARISON")
        print(f"{'='*100}")
        print(f"\n{'Iteration':<12} {'Before Output':<20} {'After Output':<20} {'Delta Output':<20} {'% Change'}")
        print("-" * 80)
        min_len = min(len(before_history), len(after_history))
        for i in range(min_len):
            b_out = before_history[i]["X"].sum()
            a_out = after_history[i]["X"].sum()
            d_out = a_out - b_out
            pct_c = d_out / b_out * 100 if b_out > 0 else 0
            print(
                f"{before_history[i]['iteration']:<12} "
                f"${b_out:>17,.2f}  ${a_out:>17,.2f}  "
                f"{d_out:>+17,.2f}   {pct_c:>+7.2f}%"
            )

    return {
        "X":          delta_X,
        "VA":         delta_VA,
        "VA_by_sector": delta_VA_by_sector,
        "intermediate": delta_int,
        "C": d_C,
        "I": d_I,
        "G": d_G,
    }
