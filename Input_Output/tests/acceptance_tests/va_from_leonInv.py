import numpy as np

'''This script seeks to evaluate whether resolving/calculating 
the value added from a given Leontief inverse is possible, and valid'''

# Example: 3-sector economy
print("=== Value Added Calculation from Leontief Inverse ===\n")

# Given: Leontief Inverse Matrix (I - A)^-1
# This is what we start with
leontief_inverse = np.array([
    [1.2, 0.3, 0.1],
    [0.2, 1.4, 0.2], 
    [0.1, 0.2, 1.3]
])

# Given: Final Demand Vector
final_demand = np.array([100, 80, 60])

print("Given Leontief Inverse Matrix (I - A)^-1:")
print(leontief_inverse)
print(f"\nGiven Final Demand: {final_demand}")

# Step 1: Calculate Total Output
total_output = leontief_inverse @ final_demand
print(f"\nStep 1 - Total Output (x = (I-A)^-1 × f):")
print(f"x = {total_output}")

# Step 2: Recover the Technical Coefficients Matrix A
# From (I - A)^-1, we need to get A
# Formula: A = I - (I - A)^-1 ^-1
I = np.eye(3)  # Identity matrix
A = I - np.linalg.inv(leontief_inverse)
print(f"\nStep 2 - Recovered Technical Coefficients Matrix A:")
print("A = I - inv((I-A)^-1)")
print(A)

# Step 3: Calculate Intermediate Consumption Matrix Z
# Z = A × x̂ (where x̂ is diagonalized total output)
x_diag = np.diag(total_output)  # Diagonalize total output
Z = A @ x_diag
print(f"\nStep 3 - Intermediate Consumption Matrix Z:")
print("Z = A × x̂")
print(Z)

# Step 4: Calculate Value Added by Sector
# Value Added = Total Output - Sum of intermediate inputs used by each sector
intermediate_inputs_used = np.sum(Z, axis=0)  # Sum columns (inputs to each sector)
value_added_by_sector = total_output - intermediate_inputs_used

print(f"\nStep 4 - Value Added Calculation:")
print(f"Intermediate inputs used by each sector: {intermediate_inputs_used}")
print(f"Value added by sector: {value_added_by_sector}")
print(f"Total value added: {np.sum(value_added_by_sector):.2f}")

# Verification: Total value added should equal total final demand in a closed economy
print(f"\nVerification:")
print(f"Total final demand: {np.sum(final_demand)}")
print(f"Total value added: {np.sum(value_added_by_sector):.2f}")
print(f"Difference (should be ~0): {abs(np.sum(final_demand) - np.sum(value_added_by_sector)):.6f}")

# Additional insight: Value-added coefficients
value_added_coefficients = value_added_by_sector / total_output
print(f"\nValue-added coefficients (value added per unit output):")
print(f"v = {value_added_coefficients}")
# Calculate value added directly using Value-added coefficients: VA = v × (I-A)^-1 × Y
total_value_added_method2 = value_added_coefficients @ leontief_inverse @ final_demand
print(f"Total Value Added (v × (I-A)^-1 × Y): {total_value_added_method2:.2f}")
print()

# Verification that using Value-added coefficients
print("VERIFICATION USING Value-added coefficients:")
print("-" * 15)
print(f"Method 1 result: {np.sum(value_added_by_sector):.2f}")
print(f"Method 2 result: {total_value_added_method2:.2f}")
print(f"Difference: {abs(np.sum(value_added_by_sector) - total_value_added_method2):.10f}")
print()

# Show the complete input-output structure
print(f"\n=== Complete Input-Output Structure ===")
print(f"Sectors: [1, 2, 3]")
print(f"Total Output: {total_output}")
print(f"Final Demand: {final_demand}")
print(f"Value Added: {value_added_by_sector}")
print(f"\nIntermediate Consumption Matrix Z:")
print(Z)

def calculate_va_coefficients(leontief_inverse, output):
    identity = np.eye(len(leontief_inverse))
    A_matrix = identity - np.linalg.inv(leontief_inverse)
    output_diag = np.diag(output)
    intermediate_inputs = np.sum(A_matrix @ output_diag, axis=0)
    value_added_by_sector = output - intermediate_inputs
    value_added_coefficients = value_added_by_sector / output
    print("\nValue Added by Sector Calculation:", value_added_by_sector)
    print("\nValue Added Coefficients: ", value_added_coefficients)
    return value_added_by_sector, value_added_coefficients

calculate_va_coefficients(leontief_inverse=leontief_inverse,output=total_output)