# Understanding Iteration and Convergence in the Input-Output Simulator

When running the tax/IO simulator, the output logs show a series of iterations where the calculated Total Output (X) changes rapidly at first, then stabilizes. This document explains the mathematical mechanics behind this behavior, using actual simulation logs as a reference.

## The Simulation Log Example

Consider the following output from a typical simulation run:

```
INFO:simulators.tax_simulator:--- Iteration 1 ---
INFO:simulators.tax_simulator:Output (X): 4250.00, Total Demand (Y): 3000.00
INFO:simulators.tax_simulator:Net Wages: 1856.25, Net Surplus: 703.12, Taxes: 440.62
...
INFO:simulators.tax_simulator:--- Iteration 2 ---
INFO:simulators.tax_simulator:Output (X): 3872.66, Total Demand (Y): 3000.00
INFO:simulators.tax_simulator:Net Wages: 1831.89, Net Surplus: 723.43, Taxes: 444.69
...
INFO:simulators.tax_simulator:--- Iteration 3 ---
INFO:simulators.tax_simulator:Output (X): 3884.50, Total Demand (Y): 3000.00
INFO:simulators.tax_simulator:Net Wages: 1832.21, Net Surplus: 723.16, Taxes: 444.63
...
INFO:simulators.tax_simulator:--- Iteration 4 ---
INFO:simulators.tax_simulator:Output (X): 3884.34, Total Demand (Y): 3000.00
INFO:simulators.tax_simulator:Net Wages: 1832.20, Net Surplus: 723.16, Taxes: 444.63
...
INFO:simulators.tax_simulator:--- Iteration 5 ---
INFO:simulators.tax_simulator:Output (X): 3884.35, Total Demand (Y): 3000.00
...
INFO:simulators.tax_simulator:--- Iteration 13 ---
INFO:simulators.tax_simulator:Output (X): 3884.35, Total Demand (Y): 3000.00
```

Notice how the Total Output `X` changes significantly between Iteration 1 and 2 (from 4250.00 down to 3872.66), then the changes become smaller, until it locks in at `3884.35` by Iteration 5. Subsequent iterations (like Iteration 13) produce the exact same result. This is not a bug; it is mathematical convergence.

## Why Does It Converge? The Supply Chain Depth

The simulator uses an iterative approach to find the total output required to satisfy a given final demand. To produce the final goods, sectors must consume intermediate goods from other sectors.

Let's look at the A matrix (technical coefficients) from the logs:

```python
A = [[0.0, 0.5, 0.0],
     [0.0, 0.0, 0.5],
     [0.0, 0.0, 0.0]]
```

This represents a supply chain:
*   Sector 2 (Bread) requires $0.50 of Sector 1 (Flour) for every $1 of output.
*   Sector 1 (Flour) requires $0.50 of Sector 0 (Wheat) for every $1 of output.

### The Iteration Breakdown

When we inject a Final Demand (Y), the economy doesn't just produce Y. It must produce Y *plus* the intermediate inputs needed to make Y.

1.  **Iteration 1:** The initial shock. The model attempts to satisfy the demand, but initially over- or under-estimates the indirect requirements before the circular flow is fully calculated.
2.  **Iteration 2:** The model calculates the first layer of intermediate inputs (e.g., the Flour needed to make the Bread).
3.  **Iteration 3:** The model calculates the second layer of indirect inputs (e.g., the Wheat needed to make the Flour that was needed to make the Bread).

Mathematically, this process is calculating the geometric series of the A matrix:

`X = Y + A·Y + A²·Y + A³·Y + ...`

Where:
*   `Y`: Final Demand
*   `A·Y`: Direct inputs needed
*   `A²·Y`: Indirect inputs (inputs for the inputs)
*   `A³·Y`: Third-level indirect inputs

### Reaching the Fixed Point

Because the technical coefficients in matrix A are fractions (less than 1), the higher powers of A become smaller and smaller. In our specific 3-sector example:

*   `A¹`: Bread needs Flour. Flour needs Wheat.
*   `A²`: Bread indirectly needs Wheat.
*   `A³`: The matrix becomes all zeros. There are no more indirect links in this short supply chain.

By Iteration 4 and 5, the model has exhausted all indirect supply chain requirements. The formula reaches a **fixed point**, mathematically equivalent to solving the Leontief Inverse:

`X = (I - A)⁻¹ · Y`

Once the iterative process reaches this fixed point (`X = 3884.35`), running further iterations (like Iteration 13) simply re-evaluates the same balanced state, resulting in no further changes. The economy has reached equilibrium.
