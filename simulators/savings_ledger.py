"""
Savings Ledger -- Investment Financing Gate
==========================================

Provides ``SavingsLedger``, a minimal aggregate cash balance used to gate
whether a capital investment (tech change with ``set_capital_requirements``)
is allowed to fire.

Design decisions (see .AGENTS/savings-ledger-spec.md):
- Single running scalar balance; no per-sector accounting.
- No overdraft: ``withdraw`` returns False and leaves balance unchanged if
  insufficient funds.
- Reserve-on-commit: the full investment cost is withdrawn at the moment the
  tech change is committed, not incrementally as it is spent.
- ``deposit`` accepts a ``source_label`` for future distinguishability
  (e.g. ``"va"`` for VA-derived savings vs ``"external"`` for injections).
"""

from __future__ import annotations

import logging
from typing import Optional

logger = logging.getLogger(__name__)


class InvestmentNotAffordableError(RuntimeError):
    """
    Raised when a capital requirement cannot be met by the current ledger balance.

    Callers that want to handle an unaffordable investment gracefully should
    catch this specific exception rather than the broad ``RuntimeError``.
    """


class SavingsLedger:
    """
    Minimal aggregate cash-balance ledger for investment financing.

    The ledger tracks a single running balance that accumulates from
    VA-derived savings (income / surplus not consumed or taxed) and any
    other explicitly deposited amounts.  It gates capital investment:
    before an investment is allowed to start, the full cost is reserved
    (withdrawn) up-front in an all-or-nothing check.

    Parameters
    ----------
    initial_balance : float
        Starting balance.  Defaults to 0.0.

    Examples
    --------
    >>> ledger = SavingsLedger()
    >>> ledger.deposit(500.0, source_label="va")
    >>> ledger.withdraw(300.0)
    True
    >>> ledger.balance
    200.0
    >>> ledger.withdraw(500.0)   # insufficient funds
    False
    >>> ledger.balance
    200.0
    """

    def __init__(self, initial_balance: float = 0.0) -> None:
        if initial_balance < 0:
            raise ValueError(
                f"initial_balance must be non-negative, got {initial_balance}"
            )
        self._balance: float = float(initial_balance)

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def balance(self) -> float:
        """Current ledger balance (read-only view)."""
        return self._balance

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def deposit(self, amount: float, source_label: str = "va") -> None:
        """
        Add ``amount`` to the ledger balance.

        Parameters
        ----------
        amount : float
            Dollar amount to deposit.  Must be non-negative.
        source_label : str
            Informational tag identifying the origin of the funds.
            Common values: ``"va"`` (VA-derived savings from the simulation
            period loop), ``"external"`` (exogenous injection / manual top-up).
            Not used in any computation today; retained for future
            distinguishability.

        Raises
        ------
        ValueError
            If ``amount`` is negative.
        """
        if amount < 0:
            raise ValueError(f"Deposit amount must be non-negative, got {amount}")
        self._balance += amount
        logger.debug(
            "SavingsLedger deposit: +%.2f [%s] -> balance=%.2f",
            amount,
            source_label,
            self._balance,
        )

    def withdraw(self, amount: float) -> bool:
        """
        Attempt to withdraw ``amount`` from the ledger.

        No overdraft is allowed.  If ``amount > balance`` the balance is
        left unchanged and ``False`` is returned.

        Parameters
        ----------
        amount : float
            Dollar amount to withdraw.  Must be non-negative.

        Returns
        -------
        bool
            ``True`` if the withdrawal succeeded; ``False`` if the balance
            was insufficient (balance is unchanged in this case).

        Raises
        ------
        ValueError
            If ``amount`` is negative.
        """
        if amount < 0:
            raise ValueError(f"Withdrawal amount must be non-negative, got {amount}")
        if amount > self._balance:
            logger.info(
                "SavingsLedger withdrawal FAILED: requested=%.2f balance=%.2f",
                amount,
                self._balance,
            )
            return False
        self._balance -= amount
        logger.debug(
            "SavingsLedger withdrawal: -%.2f -> balance=%.2f",
            amount,
            self._balance,
        )
        return True

    def can_afford(self, amount: float) -> bool:
        """Return ``True`` if the current balance covers ``amount``."""
        return self._balance >= amount

    def reset(self, balance: float = 0.0) -> None:
        """
        Reset the ledger to a given balance.

        Intended for use at the start of each simulation run so that the
        ledger lifecycle is per-run (not accumulated across UI sessions).

        Parameters
        ----------
        balance : float
            New starting balance.  Defaults to 0.0.
        """
        if balance < 0:
            raise ValueError(f"balance must be non-negative, got {balance}")
        self._balance = float(balance)
        logger.debug("SavingsLedger reset -> balance=%.2f", self._balance)

    # ------------------------------------------------------------------
    # Dunder helpers
    # ------------------------------------------------------------------

    def __repr__(self) -> str:
        return f"SavingsLedger(balance={self._balance:.2f})"
