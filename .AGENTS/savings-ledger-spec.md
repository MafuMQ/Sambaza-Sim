# Spec: Savings Ledger & Investment Financing (Phase 1)

## Context
Sambaza-Sim currently models technology change via `set_capital_requirements`:
a `requirements` dict maps sector ISIC codes/indices to dollar amounts, which
`get_capital_demand_vector` turns into a capital demand vector injected into
final demand during an "investment phase" (old A matrix, `investment_duration`
iterations), followed by a "technology phase" (new A matrix).

**Problem:** capital requirements currently fire unconditionally. There is no
check against available savings, so investment behaves like free/unconstrained
GDP rather than something financed by displaced consumption. This spec adds
the missing financing layer without changing the existing sector-goods
demand-vector mechanism.

## Design decision: keep the two concerns separate
- **Sector-level demand mechanics** (the `requirements` dict, the capital
  demand vector, injection into the Leontief solve) — **unchanged**. Do not
  restructure VA or the demand vector to be "raw cash"; that would ripple
  through the rest of the system for no real gain at this phase.
- **Affordability / financing** — new, aggregate, scalar. A single running
  cash balance that gates whether a scheduled capital requirement is allowed
  to fire at all.

The two are linked only at one point: the *total dollar sum* of a
requirements dict is what gets checked against and withdrawn from the ledger.
Which sectors that total is allocated to is irrelevant to the ledger.

## New component: `SavingsLedger`
Minimal, aggregate, cash-only.

Fields:
- `balance: float`

Methods:
- `deposit(amount: float, source_label: str = "va") -> None`
  Called each period for VA-derived savings (income/surplus not consumed or
  taxed), and separately for any exogenous injection (savings from outside
  the modeled VA — e.g. external capital, credit, manual top-up). Keeping
  `source_label` now costs nothing and avoids a later migration when we want
  to distinguish "financed by retained surplus" vs "financed by outside
  capital."
- `withdraw(amount: float) -> bool`
  Returns whether the withdrawal succeeded. **No overdraft.** If
  `amount > balance`, return `False` and leave balance unchanged.

Note: VA is *not* the only source of ledger deposits — savings can originate
elsewhere. The ledger should not assume its only inflow is the VA split.

## Financing gate: reserve-on-commit (upfront, all-or-nothing)
When an investment (a tech change with `set_capital_requirements`) is
initiated:

1. Compute `total_cost = sum(requirements.values())`.
2. Check `ledger.balance >= total_cost`.
   - If **false**: investment does not start this period. (No partial start,
     no queuing/retry logic yet — defer that.)
   - If **true**: immediately `ledger.withdraw(total_cost)` — i.e. reserve
     the full amount at commit time, not incrementally as it's spent.
3. The reserved total is then released into the capital demand vector across
   `investment_duration` iterations per the injection schedule below.

Rationale for reserve-on-commit rather than pay-as-you-go: avoids the messy
case of an investment stalling mid-way because savings dried up (which would
otherwise require deciding: pause? cancel and refund partial spend? proceed
underfunded?). Not needed yet given the current simplifying assumption that
all investments are "good" investments. It also means once we handle multiple
competing investments (deferred, see below), a second investment naturally
cannot double-spend savings already reserved by a first — no extra logic
required, because the reservation already left the balance.

## Injection schedule: even split (default)
`investment_duration` already means **iterations** (confirmed), not a
duration-with-unset-spread. Default behavior:

- Each of the `investment_duration` periods injects
  `requirements[sector] / investment_duration` for each sector in the dict.

Leave room for a future weighting vector (e.g. front-loaded for a one-off
machinery purchase, back-loaded for construction paid on completion), but
**do not build that now** — no current use case needs it.

## Explicitly deferred / out of scope for this change
- **Multi-investment competition for the same savings pool.** Only single
  investment financing is being handled now. (The reserve-on-commit design
  above is chosen partly because it will not need rework when this is
  tackled later.)
- **Cost of investment beyond the cash ledger check** (e.g. treating bad
  investments as debt-like / inflationary if they fail to pay off). Explicitly
  deferred as too complex for now — investment continues to be modeled as
  displaced consumption financed by savings, full stop, not as credit
  creation.
- **VA restructured as raw cash instead of sector-goods demand.** Would be
  "more correct" but disruptive to the existing system; the ledger absorbs
  the affordability concern instead, so this restructuring is unnecessary for
  now.
- Partial financing / underfunded investments proceeding anyway.
- Any weighting/front-loading of the injection schedule.

## Implementation checklist for the agent
1. Add `SavingsLedger` class (balance, `deposit`, `withdraw` as specified).
2. Wire ledger deposits into the existing period loop wherever VA
   income/surplus is currently computed and not consumed/taxed (source_label
   `"va"`), plus expose a way to deposit from other sources manually.
3. In the investment-initiation path (`set_capital_requirements` /
   wherever the tech change is committed to run), add the affordability
   check + reserve-on-commit withdrawal described above. Investment should
   fail to start (not silently proceed) if the check fails.
4. Implement even-split injection of the reserved total across
   `investment_duration` iterations, replacing whatever the current injection
   logic assumes about total-vs-per-period amounts (confirm today's code
   treats the dict values as per-period, and adjust so the *sum* is what's
   checked against the ledger, with per-period injection derived by dividing
   by `investment_duration`).
5. No changes to the sector-level capital demand vector mechanism, the
   Leontief solve, or the VA sector-goods structure.
