import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

import logging
from Input_Output_Model.demos.util.Setup_Data import setup_data
from Input_Output_Model.demos.util.economic_simulation import run_economic_simulation

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# --- EXECUTE ---
if __name__ == "__main__":
    # Load data from CSV files in data/ex1 directory
    print("Loading data from CSV files...")
    setup_data(source="data/ex1", overwrite_existing_data=True)
    print("\nData setup complete. Running simulation...\n")
    
    # Use a domestic good from the loaded database
    from Input_Output_Model.models.entities.Good import GoodsDatabase
    gdb = GoodsDatabase()
    goods = gdb.get_all_goods()
    
    if goods:
        # Find a non-foreign-exchange good
        domestic_goods = [g for g in goods if g.id_number != 9999]
        if domestic_goods:
            target_good = domestic_goods[0]
            print(f"Running simulation for: {target_good.name} (ISIC: {target_good.isic})")
            run_economic_simulation(target_isic=target_good.isic, demand_increase=100.0)
        else:
            print("No domestic goods found in database!")
    else:
        print("No goods found in database!")
