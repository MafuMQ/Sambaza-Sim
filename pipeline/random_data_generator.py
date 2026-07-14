import logging
import random
import time
from faker import Faker
from pipeline.db.repositories.good_repo import GoodsDatabase
from pipeline.db.repositories.supply_curve_repo import SupplyCurveDatabase
from pipeline.db.repositories.production_repo import ProductionsDatabase
from pipeline.db.repositories.tech_change_repo import TechChangeDatabase

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

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
        value_added_types = ["wages","surplus","mixed_income"]
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

def setup_random_sample_data(overwrite=True, for_custom_taxes=False):
    """Set up a random sample Input-Output model database."""
    print("="*70)
    print("Setting up sample Input-Output model database...")
    print("="*70)
    time.sleep(1)  # Sleep
    
    if overwrite:
        GoodsDatabase().clear_all_tables()
        SupplyCurveDatabase().clear_all_tables()
        ProductionsDatabase().clear_all_tables()
        TechChangeDatabase().clear_all_tables()
    
    print("\n[1/3] Creating Foreign Exchange good for imports...")
    create_foreign_exchange_good()
    
    print("\n[2/3] Creating sample goods with domestic and import production options...")
    create_sample_goods_with_imports(5, 4)
    
    print("\n[3/3] Assigning production inputs and value-added components...")
    assign_production_inputs(for_custom_taxes)
