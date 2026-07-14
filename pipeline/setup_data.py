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


def load_csv_data(csv_path, overwrite=False):
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
    
    if overwrite:
        gdb.clear_all_tables()
        scdb.clear_all_tables()
        pdb.clear_all_tables()
        from pipeline.db.repositories.tech_change_repo import TechChangeDatabase
        TechChangeDatabase().clear_all_tables()
    
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
            from pipeline.random_data_generator import setup_random_sample_data
            setup_random_sample_data(overwrite=overwrite_existing_data, for_custom_taxes=for_custom_taxes)
            
            print("\n[4/4] Evaluating the sample data...")
            evaluate_simple_data()
            print("\n" + "="*70)
            print("Database setup completed successfully!")
            print("="*70)
        else:
            logging.info(f"Loading data from source: {source}")
            load_csv_data(source, overwrite=overwrite_existing_data)
