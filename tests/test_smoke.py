import pytest
import os
import sys

# Ensure root path is in sys.path
root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if root_dir not in sys.path:
    sys.path.insert(0, root_dir)

def test_imports_and_db_models():
    """Ensure DB models and layout components can be imported without error."""
    from pipeline.db.repositories.good_repo import GoodsDatabase
    from presentation.layout import app
    assert app is not None
    
    # Check that database gives us a valid class
    db = GoodsDatabase()
    assert db is not None

def test_build_io_matrix():
    # 1) Rebuild A and VA with the new data
    from pipeline.builders.level1_matrix import build_io_matrix
    
    # 2) Actually get them. Assumes we have the data.db from the repo
    # If the database is missing, this might fail, but it expects a built DB.
    A, VA, isic_map = build_io_matrix(demoDB=False)
    
    assert A is not None
    assert VA is not None
    assert isinstance(isic_map, dict)
    assert A.shape[0] == A.shape[1]
    assert A.shape[0] == len(isic_map)

def test_simulate_shock():
    """Test running a simple simulation on the stateless IOModel."""
    from pipeline.builders.level1_matrix import build_io_matrix
    from core.io_matrix import IOModel
    import numpy as np
    
    A, VA, isic_map = build_io_matrix(demoDB=False)
    model = IOModel(A=A, VA_coeffs=VA)
    
    base_demand = np.full(model.n, 1000.0)
    X = model.simulate(base_demand)
    
    assert X is not None
    assert X.shape == (model.n,)


# ---------------------------------------------------------------------------
# SavingsLedger unit tests (no DB, pure Python)
# ---------------------------------------------------------------------------

def test_savings_ledger_basic():
    """SavingsLedger deposit, withdraw, overdraft guard, reset, can_afford."""
    from simulators.savings_ledger import SavingsLedger

    ledger = SavingsLedger()
    assert ledger.balance == 0.0

    # Deposit
    ledger.deposit(500.0, source_label="va")
    assert ledger.balance == 500.0

    ledger.deposit(200.0, source_label="external")
    assert ledger.balance == 700.0

    # Successful withdraw
    result = ledger.withdraw(300.0)
    assert result is True
    assert ledger.balance == 400.0

    # Overdraft attempt — balance must be unchanged
    result = ledger.withdraw(1000.0)
    assert result is False
    assert ledger.balance == 400.0

    # can_afford
    assert ledger.can_afford(400.0) is True
    assert ledger.can_afford(400.01) is False

    # Exact-balance withdraw (edge case)
    result = ledger.withdraw(400.0)
    assert result is True
    assert ledger.balance == 0.0

    # Reset
    ledger.deposit(100.0)
    ledger.reset()
    assert ledger.balance == 0.0

    ledger.reset(250.0)
    assert ledger.balance == 250.0


def test_savings_ledger_negative_guards():
    """SavingsLedger rejects negative deposits and withdrawals."""
    import pytest
    from simulators.savings_ledger import SavingsLedger

    ledger = SavingsLedger(initial_balance=100.0)

    with pytest.raises(ValueError):
        ledger.deposit(-1.0)

    with pytest.raises(ValueError):
        ledger.withdraw(-1.0)

    with pytest.raises(ValueError):
        SavingsLedger(initial_balance=-50.0)

    # Balance must be untouched after the failed ops
    assert ledger.balance == 100.0


def test_capital_requirements_affordability_gate():
    """set_capital_requirements raises InvestmentNotAffordableError when ledger is insufficient."""
    import pytest
    from simulators.tech_change import TechnologicalChange
    from simulators.savings_ledger import SavingsLedger, InvestmentNotAffordableError

    tech = TechnologicalChange(name="Test Investment")

    # --- Gate fires: balance too low ---
    ledger = SavingsLedger(initial_balance=100.0)
    requirements = {"sector_A": 200.0, "sector_B": 100.0}  # total = 300

    with pytest.raises(InvestmentNotAffordableError):
        tech.set_capital_requirements(requirements, investment_duration=2, ledger=ledger)

    # After failed commit, balance MUST be unchanged and requirements NOT stored
    assert ledger.balance == 100.0
    assert tech.capital_requirements == {}  # nothing was committed

    # --- Gate passes: sufficient balance ---
    ledger2 = SavingsLedger(initial_balance=500.0)
    tech2 = TechnologicalChange(name="Affordable Investment")
    tech2.set_capital_requirements(requirements, investment_duration=2, ledger=ledger2)

    # Balance should be reduced by total_cost = 300
    assert ledger2.balance == 200.0
    assert tech2.capital_requirements == requirements
    assert tech2.investment_duration == 2


def test_get_capital_demand_vector_even_split():
    """get_capital_demand_vector divides sector amounts by investment_duration."""
    import numpy as np
    from simulators.tech_change import TechnologicalChange

    tech = TechnologicalChange(name="Even Split Test")
    # Use integer indices so we don't need an isic_map lookup
    requirements = {0: 300.0, 1: 150.0}
    tech.set_capital_requirements(requirements, investment_duration=3)

    isic_map = {}  # not needed when using integer keys
    n = 4
    vec = tech.get_capital_demand_vector(isic_map, n)

    # Per-period amounts must be total / duration
    assert vec[0] == pytest.approx(100.0)   # 300 / 3
    assert vec[1] == pytest.approx(50.0)    # 150 / 3
    assert vec[2] == pytest.approx(0.0)
    assert vec[3] == pytest.approx(0.0)

    # Sum over investment_duration iterations must equal total_cost
    total_injected = vec.sum() * tech.investment_duration
    assert total_injected == pytest.approx(450.0)


def test_tax_simulator_ledger_deposits():
    """run_tax_simulation deposits unspent VA into the ledger each iteration."""
    import numpy as np
    from unittest.mock import patch
    from simulators.savings_ledger import SavingsLedger
    from core.io_matrix import IOModel

    n = 2
    A = np.zeros((n, n))   # no intermediate inputs — all output becomes VA
    VA = np.ones(n) * 1.0  # model-level VA ratio (used by IOModel)

    model = IOModel(A=A, VA_coeffs=VA)
    isic_map = {"X": 0, "Y": 1}

    base_demand = np.array([100.0, 100.0])

    # build_va_component_matrix reads from DB, so patch it with synthetic coefficients.
    # Each sector: 60% wages, 40% surplus (per $ of output).
    fake_va_coeffs = {
        "minWages":   np.array([0.0,  0.0]),
        "bonusWages": np.array([0.0,  0.0]),
        "wages":      np.array([0.6,  0.6]),
        "surplus":    np.array([0.4,  0.4]),
    }

    wage_spend = 0.8    # 20% of wages saved each period
    surplus_spend = 0.6 # 40% of surplus saved each period
    ledger = SavingsLedger()

    from simulators.tax_simulator import run_tax_simulation

    with patch("simulators.tax_simulator.build_va_component_matrix", return_value=fake_va_coeffs):
        history = run_tax_simulation(
            model=model,
            isic_map=isic_map,
            base_demand=base_demand,
            iterations=3,
            income_tax_rate=0.0,
            corporate_tax_rate=0.0,
            income_tax_applies_to="wages",
            wage_spend_rate=wage_spend,
            surplus_spend_rate=surplus_spend,
            tax_spend_rate=1.0,
            ledger=ledger,
        )

    assert len(history) == 3
    # Ledger must have accumulated savings > 0 (unspent wages + surplus over 3 iters)
    assert ledger.balance > 0.0, f"Expected positive ledger balance, got {ledger.balance}"

    # Sanity-check: X for iter 1 = Leontief solve of [100,100] with A=0 → X=[100,100]
    # total_wage_net = (0.6*100 + 0.6*100) * (1-0.0) = 120
    # total_surplus_net = (0.4*100 + 0.4*100) * (1-0.0) = 80
    # savings_iter1 = 120*(1-0.8) + 80*(1-0.6) = 24 + 32 = 56
    # (subsequent iters have different demand so exact values vary, but total > 0)
    assert ledger.balance > 0.0

    # Backward-compat: calling without ledger must not raise
    with patch("simulators.tax_simulator.build_va_component_matrix", return_value=fake_va_coeffs):
        history2 = run_tax_simulation(
            model=model,
            isic_map=isic_map,
            base_demand=base_demand,
            iterations=3,
            income_tax_rate=0.0,
            corporate_tax_rate=0.0,
            income_tax_applies_to="wages",
            wage_spend_rate=wage_spend,
            surplus_spend_rate=surplus_spend,
            tax_spend_rate=1.0,
        )
    assert len(history2) == 3



def test_execute_simulation_savings_ledger_ui():
    """Verify that execute_simulation callback correctly gates capital investments and returns full UI payload."""
    from presentation.callbacks import execute_simulation

    # 1. Gated with 0 savings -> BLOCKED
    res_blocked = execute_simulation(
        n_clicks=1, example_id=16, iterations=5, solver='supply_curves',
        inc_before=0.0, inc_after=0.0, corp_before=0.0, corp_after=0.0,
        ui_tech_changes=[], total_fd_override=None,
        wage_spend=1.0, surplus_spend=1.0, tax_spend=1.0, economy_type='open',
        initial_savings_val=0.0, savings_gate=['enforce']
    )
    # Output delta X should be 0.00 since investment was blocked
    assert res_blocked[0] == "+$0.00" or res_blocked[0] == "$0.00"
    # Investment status is Blocked
    assert "Blocked" in res_blocked[16]
    # Status banner is visible
    assert res_blocked[11].get('display') == 'block'

    # 2. Gated with 500 savings -> FINANCED
    res_financed = execute_simulation(
        n_clicks=1, example_id=16, iterations=5, solver='supply_curves',
        inc_before=0.0, inc_after=0.0, corp_before=0.0, corp_after=0.0,
        ui_tech_changes=[], total_fd_override=None,
        wage_spend=1.0, surplus_spend=1.0, tax_spend=1.0, economy_type='open',
        initial_savings_val=500.0, savings_gate=['enforce']
    )
    # Tech change is active -> output changes
    assert res_financed[0] != "$0.00"
    # Investment status is Financed & Active
    assert "Financed" in res_financed[16]
    # Ending ledger balance is $200.00 (500 - 300)
    assert res_financed[8] == "$200.00"
    # Capital requirements schedule has 2 entries (from ex 16 spec)
    assert len(res_financed[25]) == 2
    # Check charts returned
    assert res_financed[20] is not None  # trajectory figure
    assert res_financed[21] is not None  # breakdown figure


def test_two_phase_simulation_execution():
    """Verify that run_tax_simulation executes Phase 1 (capital demand injection on baseline A)
    and Phase 2 (new A_after matrix without capital injection)."""
    import numpy as np
    from unittest.mock import patch
    from core.io_matrix import IOModel
    from simulators.tech_change import TechnologicalChange
    from simulators.savings_ledger import SavingsLedger
    from simulators.tax_simulator import run_tax_simulation

    n = 2
    # Baseline A: no intermediate demand
    A_base = np.zeros((n, n))
    VA_base = np.ones(n)
    model = IOModel(A=A_base, VA_coeffs=VA_base)
    isic_map = {"SEC_0": 0, "SEC_1": 1}
    base_demand = np.array([100.0, 100.0])

    # Technological change: reduces input usage (mocked via sector change)
    # and requires $200 of capital from sector 0 over 2 iterations (i.e. $100/iter)
    tech = TechnologicalChange(name="Automation")
    tech.add_sector_change(0, "set", 0.5)
    tech.set_capital_requirements({0: 200.0}, investment_duration=2)

    fake_va_coeffs = {
        "minWages":   np.array([0.0,  0.0]),
        "bonusWages": np.array([0.0,  0.0]),
        "wages":      np.array([0.5,  0.5]),
        "surplus":    np.array([0.5,  0.5]),
    }

    with patch("simulators.tax_simulator.build_va_component_matrix", return_value=fake_va_coeffs):
        history = run_tax_simulation(
            model=model,
            isic_map=isic_map,
            base_demand=base_demand,
            iterations=4,
            tech_change=tech,
        )

    assert len(history) == 4

    # Iteration 1 (index 0): initial demand (100) + injected capital (100) = 200
    assert history[0]["phase"] == "investment"
    assert history[0]["capital_demand_total"] == pytest.approx(100.0)
    assert history[0]["Y"][0] == pytest.approx(200.0)

    # Iteration 2 (index 1): still in investment phase
    assert history[1]["phase"] == "investment"
    assert history[1]["capital_demand_total"] == pytest.approx(100.0)

    # Iteration 3 & 4: Operational phase (capital demand stops)
    for i in [2, 3]:
        h = history[i]
        assert h["phase"] == "operational"
        assert h["capital_demand_total"] == pytest.approx(0.0)


def test_execute_simulation_ui_capital_requirements():
    """Verify that execute_simulation respects custom UI capital requirements inputs without sector mapping."""
    from presentation.callbacks import execute_simulation
    from pipeline.setup_data import get_calibrated_model
    import logging

    _, isic_map = get_calibrated_model(demoDB=False, loggingLevel=logging.WARNING)
    valid_sector = list(isic_map.keys())[0]

    # Custom UI tech change with custom capital requirement
    ui_changes = [
        {
            'method': 'add_sector_change',
            'params': {'sector_idx': valid_sector, 'change_type': 'multiply', 'value': 0.8}
        }
    ]

    # Blocked: cost is 250, initial savings is 100, gate enforced
    res_blocked = execute_simulation(
        n_clicks=1, example_id=1, iterations=4, solver='supply_curves',
        inc_before=0.0, inc_after=0.0, corp_before=0.0, corp_after=0.0,
        ui_tech_changes=ui_changes, total_fd_override=None,
        wage_spend=1.0, surplus_spend=1.0, tax_spend=1.0, economy_type='open',
        initial_savings_val=100.0, savings_gate=['enforce'],
        tc_has_cap_req=['has_req'],
        tc_cap_amount=250.0, tc_cap_duration=2
    )
    assert "Blocked" in res_blocked[16]

    # Financed: cost is 250, initial savings is 300, gate enforced
    res_financed = execute_simulation(
        n_clicks=1, example_id=1, iterations=4, solver='supply_curves',
        inc_before=0.0, inc_after=0.0, corp_before=0.0, corp_after=0.0,
        ui_tech_changes=ui_changes, total_fd_override=None,
        wage_spend=1.0, surplus_spend=1.0, tax_spend=1.0, economy_type='open',
        initial_savings_val=300.0, savings_gate=['enforce'],
        tc_has_cap_req=['has_req'],
        tc_cap_amount=250.0, tc_cap_duration=2
    )
    assert "Financed" in res_financed[16]
    assert "-$250.00" in res_financed[14]
    # Schedule row shows proportional allocation
    assert len(res_financed[25]) > 0
    assert "Domestic Sectors" in res_financed[25][0]['Sector']


def test_update_controls_scenario_loading():
    """Verify that selecting a scenario populates tech changes with per-investment capital requirements."""
    from presentation.callbacks import update_controls, render_tc_changes

    # Example 16: Energy Efficiency with Capital Investment
    res16 = update_controls(16)
    loaded_changes = res16[9]

    assert len(loaded_changes) >= 1
    assert loaded_changes[0]['capital_cost'] == 300.0
    assert loaded_changes[0]['duration'] == 2

    # Render test
    rendered = render_tc_changes(loaded_changes)
    assert rendered is not None

    # Example 7: Energy efficiency without capital requirements
    res7 = update_controls(7)
    assert len(res7[9]) >= 1
    assert res7[9][0]['capital_cost'] == 0.0
    assert res7[9][0]['duration'] == 1


def test_multiple_investments_per_investment_requirements():
    """Verify that multiple investments each have their own capital requirements,
    and are gated sequentially against the savings ledger."""
    from presentation.callbacks import execute_simulation
    from pipeline.setup_data import get_calibrated_model
    import logging

    _, isic_map = get_calibrated_model(demoDB=False, loggingLevel=logging.WARNING)
    sectors = list(isic_map.keys())

    # Two distinct investments with different monetary requirements and durations
    ui_changes = [
        {
            'method': 'add_sector_change',
            'params': {'sector_idx': sectors[0], 'change_type': 'multiply', 'value': 0.8},
            'capital_cost': 100.0,
            'duration': 2,
        },
        {
            'method': 'add_sector_change',
            'params': {'sector_idx': sectors[1], 'change_type': 'multiply', 'value': 0.9},
            'capital_cost': 200.0,
            'duration': 3,
        }
    ]

    # Case A: Partial financing ($150 savings -> Inv 1 ($100) funded, Inv 2 ($200) blocked)
    res_partial = execute_simulation(
        n_clicks=1, example_id=1, iterations=4, solver='supply_curves',
        inc_before=0.0, inc_after=0.0, corp_before=0.0, corp_after=0.0,
        ui_tech_changes=ui_changes, total_fd_override=None,
        wage_spend=1.0, surplus_spend=1.0, tax_spend=1.0, economy_type='open',
        initial_savings_val=150.0, savings_gate=['enforce']
    )
    # Status is Partially Financed
    assert "Partially Financed" in res_partial[16]
    assert "-$100.00" in res_partial[14]  # Reserved $100
    assert len(res_partial[25]) == 2     # Two rows in schedule
    assert "Financed" in res_partial[25][0]['Status']
    assert "Blocked" in res_partial[25][1]['Status']
    assert res_partial[25][0]['Total_Amount'] == 100.0
    assert res_partial[25][0]['Duration'] == 2
    assert res_partial[25][1]['Total_Amount'] == 200.0
    assert res_partial[25][1]['Duration'] == 3

    # Case B: Full financing ($350 savings -> both funded)
    res_full = execute_simulation(
        n_clicks=1, example_id=1, iterations=4, solver='supply_curves',
        inc_before=0.0, inc_after=0.0, corp_before=0.0, corp_after=0.0,
        ui_tech_changes=ui_changes, total_fd_override=None,
        wage_spend=1.0, surplus_spend=1.0, tax_spend=1.0, economy_type='open',
        initial_savings_val=350.0, savings_gate=['enforce']
    )
    assert "Financed & Active" in res_full[16]
    assert "-$300.00" in res_full[14]    # Reserved $300 (100 + 200)
    assert "Financed" in res_full[25][0]['Status']
    assert "Financed" in res_full[25][1]['Status']


def test_sector_options_complete_and_sorted():
    """Verify that get_sector_options returns all available goods sorted by name."""
    from presentation.layout import get_sector_options
    options = get_sector_options()
    assert len(options) >= 5
    labels = [o['label'] for o in options]
    assert labels == sorted(labels)




