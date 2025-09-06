from Evaluators import *
from Investment import *
from sample_random_1 import *
from Demonstrators import *

if __name__ == "__main__":
    # test_setup()
    # print("Evaluation of Indices Inputs and Prices completed.\nEvaluating Indices Production Inputs to Matrix:")
    # ---
    # A_matrix, Value_Added_Vector, VA = evaluate_indicies_production_inputs_to_matrix()
    # print("\nA_matrix:")
    # print(A_matrix)
    # print("\nValue_Added_Vector:")
    # print(Value_Added_Vector)
    # print("\nVA:")
    # print(VA)
    # # print("Evaluation of Indices Production Inputs to Matrix completed.")
    # print(create_leontief_inverse(A_matrix, Value_Added=Value_Added_Vector))
    # ---

    # TODO demonstrate demand shock
    # demonstrate_demand_shock_db(np.array([40, 70, 95, 40, 75]), [10, 0, -5, 0, 0])

    # TODO demonstrate technological change
    test_technological_change(np.array([40, 70, 95, 110, 75]))
