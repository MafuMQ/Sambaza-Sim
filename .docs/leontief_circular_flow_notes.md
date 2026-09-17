# Leontief Circular Flow — Model Notes

## 1. The VA ≡ FD Identity

In the Leontief framework, VA coefficients are defined as the residual of each sector's column sum of A:

```
va_j = 1 - Σ_i a_ij
```

This means every dollar of output is entirely accounted for by intermediate inputs plus value added. It follows algebraically that:

```
VA = Σ_j va_j · X_j = Σ_j (1 - Σ_i a_ij) · X_j = X - AX = d = FD
```

**Total VA always equals total FD.** This is not a model result — it is an accounting identity that holds unconditionally. Total sectoral VA can only redistribute across sectors; it cannot grow or shrink for a given FD. The only way to change total VA is to change total FD.

> This identity holds specifically for the Leontief (matrix-inverse) solver. In the supply-curves solver, the monetary A matrix is updated each inner iteration using market prices from merit-order dispatch. Once prices deviate from their base values, the column sums of the price-adjusted A matrix no longer sum to `1 - va_j`, and the identity breaks.

---

## 2. Circular Flow Iterations

- **Iteration 1**: Uses the exogenous final demand vector supplied to `run_simulation()`.
- **Iterations 2+**: The next period's demand is the VA from the previous period, redistributed as spending:
  ```
  current_demand = C + I + G
                 = after_tax_wages + after_tax_surplus + tax_revenue
  ```
  Since VA ≡ FD, the total demand injected in iteration 2 is identical to iteration 1.

### When iterations produce dynamics

Because VA ≡ FD, multi-period dynamics can only arise from changes in the **sectoral composition** of demand, not the total. Conditions that cause this:

| Condition | Effect |
|---|---|
| `consumption_rate < 1.0` | Leakage — total demand shrinks each iteration |
| C/I/G sector proportions differ from initial FD proportions | Sector mix of demand shifts each round |
| Taxes present (income/corporate) | C, I, G split differently than wages/surplus, affecting sector mix if their proportions differ |
| Supply-curves solver | Price-adjusted A matrix breaks the strict VA = FD identity |

Without any of these conditions (Leontief, no taxes, `consumption_rate=1.0`, proportional distribution matching initial FD), iteration 1 is immediately the fixed point and all subsequent iterations are identical.

---

## 3. Technology Change and Sectoral VA

### Example 7 setup

All observations in this section refer to Example 7 from the demo: a 30% reduction in energy-proxy (Sector 4, `A6178_413_70647`) inputs across all sectors. The economy has 5 domestic sectors and a fixed final demand of $1,000.

**Baseline technical coefficients (column = sector being produced, row = input required):**
```
         Sec0    Sec1    Sec2    Sec3    Sec4
Sec0 |  0.0000  0.1897  0.0000  0.0323  0.0000
Sec1 |  0.2323  0.0000  0.0000  0.1935  0.1333
Sec2 |  0.1111  0.0000  0.0000  0.1613  0.0000
Sec3 |  0.0606  0.0000  0.0700  0.0000  0.3222   ← Sec3 heavily supplies Sec4
Sec4 |  0.1111  0.0862  0.3500  0.0000  0.0000   ← row affected by tech change
```
**Baseline VA coefficients:** `[0.4848, 0.7241, 0.5800, 0.6129, 0.5444]`

**Tech change — multiply all Sec4 input coefficients by 0.7 (−30%):**
```
ΔA row 4: Sec0 −0.0333  Sec1 −0.0259  Sec2 −0.1050  (Sec3, Sec4 unchanged)
ΔVA:      Sec0 +0.0333  Sec1 +0.0259  Sec2 +0.1050  Sec5 +1.0000
```
After change, VA coefficients: `[0.5182, 0.7500, 0.6850, 0.6129, 0.5444]`

When a tech change reduces intermediate input requirements, two distinct mechanisms affect sectoral VA:

**Mechanism 1 — VA coefficient increase (supply side):**  
Sectors that previously bought the reduced input now have a lower column sum of A. Since `va_j = 1 - Σ_i a_ij`, their VA coefficient rises mechanically. With the same output level, they retain more of each dollar of revenue as VA.

**Mechanism 2 — Output reduction (demand side):**  
The sector whose output is now less demanded as an intermediate input sees its required gross output fall. Since VA = va_j × X_j, lower output directly reduces that sector's absolute VA — even if its own VA coefficient is unchanged.

### Sectoral results

```
Sector               Output (X)   Value Added  Intermediate   ΔOutput    ΔVA
---------------------------------------------------------------------------------
Baseline:
[2] A2327_978_13        384.26       222.87       161.39
[4] A6178_413_70647     303.48       165.23       138.25
[3] A3790_132_63        294.71       180.63       114.08
[1] A2227_974_71103     374.29       271.03       103.25
[0] A1397_975_51074     330.49       160.24       170.25
TOTAL                 1687.23      1000.00       687.23

After tech change:
[2] A2327_978_13        380.56       260.68       119.88      -3.70   +37.81
[4] A6178_413_70647     240.49       130.93       109.56     -62.99   -34.29
[3] A3790_132_63        273.97       167.92       106.05     -20.75   -12.72
[1] A2227_974_71103     361.14       270.85        90.28     -13.15    -0.18
[0] A1397_975_51074     327.33       169.62       157.71      -3.16    +9.38
TOTAL                 1583.48      1000.00       583.48    -103.74    -0.00
```

| Sector | ΔVA | Mechanism |
|---|---|---|
| Sec 2 | +37.81 | Highest energy input (0.35 → 0.245); largest VA coefficient gain; output barely changes |
| Sec 0 | +9.38 | Smaller energy input (0.111 → 0.078); moderate VA coefficient gain |
| Sec 1 | −0.18 | Small energy input (0.086 → 0.060); VA coefficient gain nearly offset by output fall |
| Sec 4 (energy) | −34.29 | Its output is demanded less across all sectors; gross output falls 20.76% |
| Sec 3 | −12.72 | Indirect — supplies Sec 4 with coeff 0.3222; Sec 4 output fall reduces Sec 3 demand |

Total VA remains $1,000 — gains and losses sum to zero because FD is held constant.

The **meaningful economic signal** is the gross output reduction (−$103.74, −6.15%): the same final basket of goods is produced with less total intermediate activity, reflecting genuine efficiency improvement.

---

## 4. Extending the Model for Growth

Because VA ≡ FD unconditionally, efficiency gains do not generate new income — they only redistribute it across sectors. Intermediate costs are payments from one sector to another sector already inside the economy; a reduction in one sector's intermediate purchases is simultaneously a reduction in another sector's revenue and VA. There is no net resource freed up.

Total VA (= FD) can only grow through external injections — spending that enters the circular flow from outside the closed system: export demand, government deficit spending, foreign investment, or new household borrowing. Without such an injection, the efficiency gain is real (less gross output required for the same final basket) but the nominal economy does not grow.

---

## 5. What VA ≡ FD Implies About Prices and Savings

VA ≡ FD holds unconditionally in all cases. The question is: given a technology improvement, at what **level** do VA and FD settle, and what does that imply about prices and income in the real world?

### Case A — Prices unchanged, producers retain the gain
The cost reduction becomes additional income for workers and owners. The sector's VA coefficient rises. VA (= FD) remains at the original level because all of the additional income is spent as demand. No savings, no price change — the gain goes entirely to producer income.

### Case B — Prices fall, consumers maintain nominal spending
Consumers face lower prices but continue spending $x on the sector's goods, receiving more quantity for the same outlay. VA (= FD) remains at the original level. Consumers capture the gain as **real quantity**, not as monetary savings.

### Case C — Prices fall, consumers maintain real quantity
Consumers buy the same physical quantity at lower prices, spending less nominally. FD falls, and since VA ≡ FD, VA falls by the same amount. They settle at a new, lower equilibrium level. The reduction from the original level represents genuine **consumer savings** — a leakage from the circular flow.

### Summary

| Scenario | Prices | VA = FD level | Who captures the gain |
|---|---|---|---|
| Producers retain gain | Unchanged | Same as before | Workers / owners (higher income) |
| Consumers buy more quantity | Fall | Same as before | Consumers (more real goods, same nominal spend) |
| Consumers save | Fall | Lower than before | Consumers (same real goods, less nominal spend) |

In all three cases VA = FD. The cases differ in the **level** at which the identity holds.

The Leontief solver, being purely nominal with no explicit price mechanism, cannot distinguish between Cases A and B — both produce the same numerical result. Case C requires either an explicit price model (supply-curves solver) or a `consumption_rate < 1.0` to represent the leakage as reduced nominal FD feeding into the next iteration.

