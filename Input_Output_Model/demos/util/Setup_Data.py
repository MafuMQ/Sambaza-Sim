from __future__ import annotations
import logging
from Input_Output_Model.util.Evaluators import *
from Input_Output_Model.models.entities.Good import GoodsDatabase
from Input_Output_Model.models.entities.SupplyCurve import SupplyCurveDatabase
from Input_Output_Model.models.entities.Production import ProductionsDatabase
import random
from pathlib import Path
import time

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

from faker import Faker
fake = Faker()

# ISIC code for foreign exchange/money used in imports
FOREIGN_EXCHANGE_ISIC = "A9999_999_999"

def create_foreign_exchange_good():
    """Create a special 'Foreign Exchange' good to represent import costs."""
    gdb = GoodsDatabase()
    scdb = SupplyCurveDatabase()
    
    # Check if it already exists
    existing_goods = gdb.get_all_goods()
    for good in existing_goods:
        if good.isic == FOREIGN_EXCHANGE_ISIC: # pyright: ignore[reportGeneralTypeIssues]
            logger.info(f"Foreign Exchange good already exists")
            return
    
    # Create the foreign exchange good
    gdb.add_good(
        name="Foreign Exchange",
        descriptive_name="Represents foreign currency/money used to purchase imports",
        id_number=9999,
        isic=FOREIGN_EXCHANGE_ISIC,

    )
    
    # Create its supply curve with a fixed price (e.g., exchange rate = 1.0)
    scdb.add_supply_curve(
        name="Foreign Exchange",
        id_number=9999,
        isic=FOREIGN_EXCHANGE_ISIC
    )
    
    logger.info(f"Created Foreign Exchange good with ISIC: {FOREIGN_EXCHANGE_ISIC}")

def add_import_production(good_name, produce, isic):
    production_db = ProductionsDatabase()

    # Create a production entry for the imported good
    # Use a unique id_number by combining 9 prefix with produce id
    import_id = int("9"+str(produce))
    
    # Import cost in foreign exchange
    import_cost = fake.random_int(min=100, max=1000)
    
    production_db.add_production(
        name="IMPORT",
        id_number=import_id,
        isic=isic,
        producer=99999,
        produce=produce,
        produce_name=good_name,
        production_inputs={FOREIGN_EXCHANGE_ISIC: import_cost},  # Uses foreign exchange to "purchase" imports
        production_added_values={"tariffs": fake.random_int(min=0, max=50)},  # Optional: tariffs/duties as value added
        production_rate=0,  # Unlimited production capacity
        production_quantity=-1,
        price=import_cost + fake.random_int(min=0, max=50)  # Cost + tariffs/margins
    )
    logger.info(f"Added IMPORT production for {good_name} with id_number={import_id}")

def test_good(n=5, pn=5):
    """Create test goods with domestic productions and import options."""
    gdb = GoodsDatabase()
    scdb = SupplyCurveDatabase()
    
    for _ in range(n):
        name = fake.word().capitalize() + " Good"
        desc = fake.sentence()
        id_number = fake.unique.random_int(min=2000, max=2999)
        isic = fake.bothify(text='A####_###_' + ''.join(fake.random_choices(elements='0123456789', length=fake.random_int(min=1, max=5))))
        gdb.add_good(name=name, descriptive_name=desc, id_number=id_number, isic=isic)
        logger.info(f"Adding IMPORT production for good: {name} (id_number={id_number}, isic={isic})")
        add_import_production(name, id_number, isic)
        scdb.add_supply_curve(name=name, id_number=id_number, isic=isic)  # each new good has a supply curve
        test_production_with_args(produce=id_number, isic=isic, n=pn)  # add numbered random local productions for the good

def test_production_with_args(produce, isic, n=5):
    """Create test production methods for a given good."""
    pdb = ProductionsDatabase()
    
    for _ in range(n):
        name = fake.word().capitalize() + " Production"
        id_number = fake.unique.random_int(min=4000, max=4999)
        producer = fake.random_int(min=1000, max=9999)
        produce_name = fake.word().capitalize() + " Good"

        # Production fields
        production_inputs = fake.json()  # Random JSON for production inputs
        production_added_values = fake.json()
        production_rate = fake.random_int(min=1, max=100)
        production_quantity = fake.random_int(min=1, max=100)
        price = fake.random_int(min=5, max=5000)
        
        pdb.add_production(
            name=name, 
            id_number=id_number, 
            isic=isic, 
            producer=producer, 
            produce=produce, 
            produce_name=produce_name,
            production_inputs=production_inputs, 
            production_added_values=production_added_values,
            production_rate=production_rate, 
            production_quantity=production_quantity,
            price=price
        )

def test_production_inputs():
    pdb = ProductionsDatabase()
    gdb = GoodsDatabase()
    existing_goods = gdb.get_all_goods()
    
    # Exclude Foreign Exchange from being used as a production input for domestic production
    existing_goods_isic = [good.isic for good in existing_goods if good.isic != FOREIGN_EXCHANGE_ISIC] # pyright: ignore[reportGeneralTypeIssues]
    
    value_added_types = ["wages","surplus","taxes","mixed_income"]
    productions = pdb.get_all_productions()
    
    logger.info(f"Updating production inputs for {len(productions)} productions")
    import_count = 0
    updated_count = 0
    
    for production in productions:
        if production.name != "IMPORT":  # pyright: ignore[reportGeneralTypeIssues] # Skip IMPORT productions 
            # Create realistic inputs: use 2-4 random goods (not all goods, and not the good itself)
            # Generate MONETARY costs: dollar value of each input needed per production run
            available_inputs = [isic for isic in existing_goods_isic if isic != production.isic] # pyright: ignore[reportGeneralTypeIssues]
            num_inputs = min(fake.random_int(min=2, max=4), len(available_inputs))
            selected_inputs = random.sample(available_inputs, num_inputs) if available_inputs else []
            
            # Monetary input costs (dollar value of input materials needed)
            inputs = {isic: fake.random_int(min=1, max=50) for isic in selected_inputs}
            # Monetary value added components (dollar value of labor, capital, etc.)
            added_values = {value_type: fake.random_int(min=1, max=20) for value_type in value_added_types}
            pdb.update_production(int(production.id), production_inputs=inputs, production_added_values=added_values)  # pyright: ignore[reportArgumentType]
            updated_count += 1
        else:
            import_count += 1
            logger.info(f"Skipping IMPORT production: id={production.id}, id_number={production.id_number}, produce={production.produce}")
    
    logger.info(f"Updated {updated_count} regular productions, skipped {import_count} IMPORT productions")

def handle_existing_db_files(do_what="remove", remove_what=["data.db", "dataDEMO.db"]):
    repo_root = Path(__file__).resolve().parents[3]  # go up from Input_Output/demos/util -> repo root
    for fname in remove_what:
        target = repo_root / fname
        if target.exists():
            if do_what == "remove":
                print("Database file exists, removing to allow fresh setup:", target)
                try:
                    target.unlink()
                    logging.info(f"Removed existing database file: {target}")
                except Exception as e:
                    logging.warning(f"Could not remove {target}: {e}")
            elif do_what == "warn":
                logging.warning(f"Database file already exists: {target}. Setup may fail if data conflicts.")
            elif do_what == "ignore":
                logging.info(f"Database file exists, but ignoring as per configuration: {target}")
                return False
            # else do nothing
    return True

def setup_random_sample_data(ignore_if_exists=False):
    """Set up random sample data for testing the Input-Output model."""
    do_what = "remove"
    if ignore_if_exists:
        do_what = "ignore"
    if handle_existing_db_files(do_what=do_what, remove_what=["data.db", "dataDEMO.db"]):
        print("Setting up random sample data...")
        time.sleep(1)  # Sleep
        print("Creating Foreign Exchange good for imports:")
        create_foreign_exchange_good()
        print("\nTesting Good:")
        test_good(5, 4)
        print("\nTesting Production inputs:")
        test_production_inputs()
        print("\nTesting completed.")
        print("\nEvaluating ISIC codes:")
        evaluate_goods_isic()
        print("Evaluation of ISIC codes completed.\nEvaluating Production Prices:")
        evaluate_productions_price()
        print("\nEvaluation of Production prices completed.\nBuilding Supply Curves:")
        build_supply_curves(with_functions=True)
        print("Supply curves built.\nBuilding Input-Output Matrix:")
        build_io_matrix()
        print("Input-Output matrix built successfully.")