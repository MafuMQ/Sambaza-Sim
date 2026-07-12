from __future__ import annotations
import logging
from pipeline.builders.level3_productions import evaluate_productions_price
from pipeline.builders.level2_curves import build_supply_curves
from pipeline.builders.level1_matrix import build_io_matrix
from core.isic_utils import evaluate_goods_isic
from pipeline.db.repositories.good_repo import GoodsDatabase
from pipeline.db.repositories.supply_curve_repo import SupplyCurveDatabase
from pipeline.db.repositories.production_repo import ProductionsDatabase
from pipeline.db.repositories.tech_change_repo import TechChangeDatabase
import random
from pathlib import Path
import time
import csv
import json

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

from core.io_matrix import IOModel

def get_calibrated_model(demoDB: bool = False, loggingLevel: int = logging.WARNING) -> tuple[IOModel, dict]:
    """
    Acts as the strict gatekeeper: loads data from DB, formats it into standardized NumPy arrays,
    and returns a pre-calibrated stateless IOModel along with the ISIC map.
    """
    A_mon, VA_mon, isic_map = build_io_matrix(demoDB=demoDB, loggingLevel=loggingLevel)
    model = IOModel(A=A_mon, VA_coeffs=VA_mon)
    return model, isic_map

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
        if good.isic == FOREIGN_EXCHANGE_ISIC:
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

def create_sample_goods_with_imports(n=5, pn=5):
    """Create sample goods with domestic production methods and import options."""
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
        create_domestic_production_methods(produce=id_number, isic=isic, n=pn)  # add numbered random local productions for the good

def create_domestic_production_methods(produce, isic, n=5):
    """Create domestic production methods for a given good."""
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

def handle_added_values():
    # Implement logic to handle added values
    minWages = fake.random_int(min=5, max=15)
    bonusWages = fake.random_int(min=0, max=10)
    wages = minWages + bonusWages
    surplus = fake.random_int(min=5, max=40)
    return {"minWages": minWages, "bonusWages": bonusWages, "wages": wages, "surplus": surplus}

def assign_production_inputs(for_custom_taxes=False):
    """Assign realistic production inputs and value-added components to all domestic productions."""
    pdb = ProductionsDatabase()
    gdb = GoodsDatabase()
    existing_goods = gdb.get_all_goods()
    
    # Exclude Foreign Exchange from being used as a production input for domestic production
    existing_goods_isic = [good.isic for good in existing_goods if good.isic != FOREIGN_EXCHANGE_ISIC]
    
    if not for_custom_taxes:
        value_added_types = ["wages","surplus","taxes","mixed_income"]
    else:
        value_added_types = ["minWages","bonusWages","wages","surplus"] # for reference, this will be ignored
    productions = pdb.get_all_productions()
    
    logger.info(f"Updating production inputs for {len(productions)} productions")
    import_count = 0
    updated_count = 0
    
    for production in productions:
        if production.name != "IMPORT":  # pyright: ignore[reportGeneralTypeIssues] # Skip IMPORT productions 
            # Create realistic inputs: use 2-4 random goods (not all goods, and not the good itself)
            # Generate MONETARY costs: dollar value of each input needed per production run
            available_inputs = [isic for isic in existing_goods_isic if isic != production.isic]
            num_inputs = min(fake.random_int(min=2, max=4), len(available_inputs))
            selected_inputs = random.sample(available_inputs, num_inputs) if available_inputs else []
            
            # Monetary input costs (dollar value of input materials needed)
            inputs = {isic: fake.random_int(min=1, max=50) for isic in selected_inputs}
            # Monetary value added components (dollar value of labor, capital, etc.)
            if  not for_custom_taxes:
                added_values = {value_type: fake.random_int(min=1, max=20) for value_type in value_added_types}
            else:
                added_values = handle_added_values()
            pdb.update_production(int(production.id), production_inputs=inputs, production_added_values=added_values)  # pyright: ignore[reportArgumentType]
            updated_count += 1
        else:
            import_count += 1
            logger.info(f"Skipping IMPORT production: id={production.id}, id_number={production.id_number}, produce={production.produce}")
    
    logger.info(f"Updated {updated_count} regular productions, skipped {import_count} IMPORT productions")

def load_tech_changes_if_available(source_dir: str) -> int:
    """
    Load tech change configurations from CSV files if they exist.
    Checks both the source_dir and the parent 'data/' directory.
    
    Parameters:
    -----------
    source_dir : str
        Source directory containing CSV files (e.g., "data/ex2")
    
    Returns:
    --------
    int : Number of tech changes loaded
    """
    from simulators.tech_change_loader import load_tech_changes_from_csv
    
    source_path = Path(source_dir)
    
    # Check in source directory first
    tax_policy_csv = source_path / "tax_policies.csv"
    tech_change_csv = source_path / "tech_changes.csv"
    
    # Fallback to parent data/ directory with alternative names
    if not tax_policy_csv.exists():
        tax_policy_csv = Path("data") / "tax_policy_examples.csv"
    if not tech_change_csv.exists():
        tech_change_csv = Path("data") / "tech_change_examples.csv"
    
    # Check if either CSV exists
    if not tax_policy_csv.exists() and not tech_change_csv.exists():
        logger.info("No tech change CSV files found, skipping tech change loading")
        return 0
    
    try:
        count = load_tech_changes_from_csv(
            tax_policy_csv=str(tax_policy_csv) if tax_policy_csv.exists() else None,
            tech_change_csv=str(tech_change_csv) if tech_change_csv.exists() else None,
            database_url="sqlite:///data.db",
            clear_existing=True
        )
        logger.info(f"Loaded {count} tech change configurations")
        return count
    except Exception as e:
        logger.error(f"Failed to load tech changes: {e}")
        return 0

def handle_existing_db_files(do_what="remove", remove_what=["data.db", "dataDEMO.db"]):
    repo_root = Path(__file__).resolve().parents[1]  # go up from pipeline -> repo root
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
                logging.info(f"Database file already exists and will NOT be removed: {target}")
    return True

def evaluate_simple_data():
        print("\n[1/4] Validating ISIC codes...")
        evaluate_goods_isic()
        
        print("\n[2/4] Validating production prices...")
        evaluate_productions_price()
        
        print("\n[3/4] Building supply curves...")
        build_supply_curves(with_functions=True)
        
        print("\n[4/4] Building Input-Output matrix...")
        build_io_matrix()

def setup_random_sample_data(overwrite=True, for_custom_taxes=False):
    """Set up a random sample Input-Output model database."""
    print("="*70)
    print("Setting up sample Input-Output model database...")
    print("="*70)
    time.sleep(1)  # Sleep
    
    print("\n[1/4] Creating Foreign Exchange good for imports...")
    create_foreign_exchange_good()
    
    print("\n[2/4] Creating sample goods with domestic and import production options...")
    create_sample_goods_with_imports(5, 4)
    
    print("\n[3/4] Assigning production inputs and value-added components...")
    assign_production_inputs(for_custom_taxes)
    
    print("\n[4/4] Evaluating the sample data...")
    evaluate_simple_data()

    print("\n" + "="*70)
    print("Database setup completed successfully!")
    print("="*70)

def load_csv_data(csv_path):
    """Load goods and productions data from CSV files in the specified path."""
    print("="*70)
    print(f"Loading data from CSV files at: {csv_path}")
    print("="*70)
    time.sleep(1)
    
    csv_path = Path(csv_path)
    goods_csv = csv_path / "goods.csv"
    productions_csv = csv_path / "productions.csv"
    
    # Validate CSV files exist
    if not goods_csv.exists():
        raise FileNotFoundError(f"Goods CSV file not found at: {goods_csv}")
    if not productions_csv.exists():
        raise FileNotFoundError(f"Productions CSV file not found at: {productions_csv}")
    
    gdb = GoodsDatabase()
    scdb = SupplyCurveDatabase()
    pdb = ProductionsDatabase()
    
    # Load goods
    print("\n[1/4] Loading goods from CSV...")
    goods_count = 0
    with open(goods_csv, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            gdb.add_good(
                name=row['name'],
                descriptive_name=row['descriptive_name'],
                id_number=int(row['id_number']),
                isic=row['isic']
            )
            # Create supply curve for each good
            scdb.add_supply_curve(
                name=row['name'],
                id_number=int(row['id_number']),
                isic=row['isic']
            )
            goods_count += 1
    logger.info(f"Loaded {goods_count} goods from CSV")
    
    # Load productions
    print("\n[2/4] Loading productions from CSV...")
    productions_count = 0
    with open(productions_csv, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            # Parse JSON fields for production_inputs and production_added_values
            production_inputs = json.loads(row['production_inputs']) if row['production_inputs'] else {}
            production_added_values = json.loads(row['production_added_values']) if row['production_added_values'] else {}
            
            pdb.add_production(
                name=row['name'],
                id_number=int(row['id_number']),
                isic=row['isic'],
                producer=int(row['producer']),
                produce=int(row['produce']),
                produce_name=row['produce_name'],
                production_inputs=production_inputs,
                production_added_values=production_added_values,
                production_rate=int(row['production_rate']) if row['production_rate'] else 0,
                production_quantity=int(row['production_quantity']) if row['production_quantity'] else -1,
                price=int(row['price']) if row['price'] else 0
            )
            productions_count += 1
    logger.info(f"Loaded {productions_count} productions from CSV")
    
    # Evaluate the loaded data
    print("\n[3/4] Evaluating loaded data...")
    evaluate_simple_data()
    
    # Load tech changes (if CSV files exist)
    print("\n[4/4] Loading tech change configurations...")
    tech_changes_loaded = load_tech_changes_if_available(csv_path)
    
    print("\n" + "="*70)
    print(f"Successfully loaded {goods_count} goods and {productions_count} productions!")
    if tech_changes_loaded > 0:
        print(f"Successfully loaded {tech_changes_loaded} tech change configurations!")
    print("="*70)

def setup_data(source=None, overwrite_existing_data=False, for_custom_taxes=False, logging_level=logging.INFO):
    """Set up data from a specified source or create random sample data."""
    logging.getLogger().setLevel(logging_level)
    do_what = "remove"
    if not overwrite_existing_data:
        do_what = "ignore"
    if handle_existing_db_files(do_what=do_what, remove_what=["data.db", "dataDEMO.db"]): #if files handled successfully
        if source is None:
            setup_random_sample_data(overwrite=overwrite_existing_data, for_custom_taxes=for_custom_taxes)
        else:
            logging.info(f"Loading data from source: {source}")
            load_csv_data(source)


