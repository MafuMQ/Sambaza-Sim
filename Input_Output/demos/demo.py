import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

from Input_Output.util.Evaluators import *
from Input_Output.models.entities.Investment import *
from Input_Output.demos.util.Setup_Data import *
from Input_Output.demos.util.Demonstrators import *

if __name__ == "__main__":
    setup_random_sample_data()
    print("Evaluation of Indices Inputs and Prices completed.\nEvaluating Indices Production Inputs to Matrix:")
    # ---
    A_matrix, Value_Added_Vector, VA = evaluate_indicies_production_inputs_to_matrix()
    print("\nA_matrix:")
    print(A_matrix)
    print("\nValue_Added_Vector:")
    print(Value_Added_Vector)
    print("\nVA:")
    print(VA)
    # print("Evaluation of Indices Production Inputs to Matrix completed.")
    print(create_leontief_inverse(A_matrix, Value_Added=Value_Added_Vector))
    # ---

    # TODO demonstrate demand shock
    # demonstrate_demand_shock_db(np.array([40, 70, 95, 40, 75]), [10, 0, -5, 0, 0])

    # TODO demonstrate technological change
    demonstrate_technological_change(np.array([40, 70, 95, 110, 75]))

    # TODO demonstrate technological improvement
    # demonstrate_technological_improvement(np.array([40, 70, 95, 110, 75]))