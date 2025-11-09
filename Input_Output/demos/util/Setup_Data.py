from __future__ import annotations
import logging
from Input_Output.util.Evaluators import *
from Input_Output.models.entities.Investment import *
from Input_Output.models.entities.Producer import ProducersDatabase
from Input_Output.models.entities.Good import GoodsDatabase
from Input_Output.models.entities.Good_Indice import GoodsIndiceDatabase
from Input_Output.models.entities.Production import ProductionsDatabase
import random
import json

logging.basicConfig(level=logging.INFO)

from faker import Faker
fake = Faker()

def add_import_producion(good_name, produce,isic):
    production_db = ProductionsDatabase()

    # Create a production entry for the imported good
    production_db.add_production(
        name="IMPORT",
        id_number=int("9"+str(produce)),
        isic=isic,
        producer=99999,
        produce=produce,
        produce_name=good_name,
        production_inputs={"A9999_999_999": 1000},  # Assuming the good is imported from a specific ISIC code
        production_added_value=0,
        production_rate=0, # Unlimited production capacity
        price = fake.random_int(min=5, max=5000)
)

def test_producer(n=5):
    pdb = ProducersDatabase()
    for _ in range(n):
        name = fake.company()
        desc = fake.catch_phrase()
        id_number = fake.unique.random_int(min=1000, max=9999)
        
        # Contact fields
        contact_name = fake.name()
        contact_email = fake.email()
        contact_phone = fake.phone_number()
        contact_phone2 = fake.phone_number()  # Optional second phone number
        contact_website = fake.url()
        # Address fields
        address = fake.address()
        address_street = fake.street_address()  # Optional street address
        address_city = fake.city()
        address_country = fake.country()
        address_postal_code = fake.postalcode()

        # pdb.add_producer(name=name, descriptive_name=desc, id_number=id_number)

        pdb.add_producer(name=name, descriptive_name=desc, id_number=id_number, 
                    contact_name=contact_name,contact_email=contact_email, 
                    contact_phone=contact_phone, contact_phone2=contact_phone2,
                    contact_website=contact_website,
                    address=address,address_street=address_street,
                    address_city=address_city,address_country=address_country,
                    address_postal_code=address_postal_code)

def test_good(n=5,pn=5):
    gdb = GoodsDatabase()
    for _ in range(n):
        name = fake.word().capitalize() + " Good"
        desc = fake.sentence()
        id_number = fake.unique.random_int(min=2000, max=2999)
        isic = fake.bothify(text='A####_###_' + ''.join(fake.random_choices(elements='0123456789', length=fake.random_int(min=1, max=5))))
        gdb.add_good(name=name, descriptive_name=desc, id_number=id_number, isic=isic)
        add_import_producion(name, id_number,isic)
        GoodsIndiceDatabase().add_good_indice(name=name,id_number=id_number,isic=isic,quantity=0) # each new good has an index
        test_production_with_args(produce=id_number, isic=isic, n=pn) # we add numbered random local productions for the good

def test_good_indice(n=5):
    gidb = GoodsIndiceDatabase()
    for _ in range(n):
        name = fake.word().capitalize() + " Indice"
        id_number = fake.unique.random_int(min=3000, max=3999)
        isic = fake.bothify(text='A####_###_' + ''.join(fake.random_choices(elements='0123456789', length=fake.random_int(min=1, max=5))))
        quantity = fake.random_int(min=1, max=1000)
        gidb.add_good_indice(name=name, id_number=id_number, isic=isic, quantity=quantity)

def test_production(n=5):
    id_number = fake.unique.random_int(min=4000, max=4999)
    isic = fake.bothify(text='A####_###_' + ''.join(fake.random_choices(elements='0123456789', length=fake.random_int(min=1, max=5))))
    test_production_with_args(produce=fake.unique.random_int(min=2000, max=2999), isic=isic, n=n)

def test_production_with_args(produce, isic, n=5):
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
        production_material_efficiency = fake.random_int(min=1, max=100)
        production_labour_efficiency = fake.random_int(min=1, max=100)
        production_energy_efficiency = fake.random_int(min=1, max=100)

        # Contact fields
        contact_name = fake.name()
        contact_email = fake.email()
        contact_phone = fake.phone_number()
        contact_phone2 = fake.phone_number()  # Optional second phone number
        contact_website = fake.url()
        # Address fields
        address = fake.address()
        address_street = fake.street_address()  # Optional street address
        address_city = fake.city()
        address_country = fake.country()
        address_postal_code = fake.postalcode()

        price = fake.random_int(min=5, max=5000)
        
        # pdb.add_production(name=name, id_number=id_number, producer=producer, produce=produce, produce_name=produce_name)
        pdb.add_production(
            name=name, id_number=id_number, isic=isic, producer=producer, produce=produce, produce_name=produce_name,
            production_inputs=production_inputs, production_added_values=production_added_values,
            production_rate=production_rate, production_material_efficiency=production_material_efficiency,
            production_labour_efficiency=production_labour_efficiency, production_energy_efficiency=production_energy_efficiency,
            contact_name=contact_name, contact_email=contact_email, contact_phone=contact_phone,
            contact_phone2=contact_phone2, contact_website=contact_website,
            address=address, address_street=address_street, address_city=address_city,
            address_country=address_country, address_postal_code=address_postal_code, price=price
        )

def test_production_inputs():
    pdb = ProductionsDatabase()
    gdb = GoodsDatabase()
    existing_goods = gdb.get_all_goods()
    existing_goods_isic = [good.isic for good in existing_goods]
    value_added_types = ["wages","surplus","taxes","mixed_income"]
    productions = pdb.get_all_productions()
    for production in productions:
        if production.name != "IMPORT":  # pyright: ignore[reportGeneralTypeIssues] # Skip IMPORT productions 
            # inputs = {random.choice(existing_goods_isic): fake.random_int(min=1, max=100) for _ in range(3)} looks more realistic if there isn't a row/column with only zeros
            inputs = {isic: fake.random_int(min=1, max=100) for isic in existing_goods_isic}
            added_values = {value_type: fake.random_int(min=1, max=50) for value_type in value_added_types}
            pdb.update_production(int(production.id), production_inputs=inputs, production_added_values=added_values)  # pyright: ignore[reportArgumentType]

def test_create_indice_investment_random(produce_isic:str) -> Investment:
    gdb = GoodsDatabase()
    existing_goods = gdb.get_all_goods()
    existing_goods_isic = [good.isic for good in existing_goods]
    value_added_types = ["wages","surplus","taxes","mixed_income"]
    # inputs = {random.choice(existing_goods_isic): fake.random_int(min=1, max=100) for _ in range(3)} looks more realistic if there isn't a row/column with only zeros
    inputs = {isic: fake.random_int(min=1, max=100) for isic in existing_goods_isic}
    added_values = {value_type: fake.random_int(min=1, max=50) for value_type in value_added_types}
    investment = Investment(id=fake.random_int(min=1, max=1000), name=fake.word(), type_of_investment=InvestmentType.INDICE, produce_isic=produce_isic, implementation_cost=fake.random_number(digits=5))
    investment.set_investment_metrics(inputs, added_values)
    # investment.price_history = indice.price_history
    # investment.quantity = indice.quantity

    return investment

def test_create_indice_investment(
    produce_isic: str, 
    improvement_percentage: float = 0.0,
    improvement_type: str = "total_cost", 
    target_va_type: str | None = None,
    investment_cost: float | None = None
) -> Investment:
    """
    Create an investment with optional improvements/deteriorations based on EXISTING indice values.
    
    Args:
        produce_isic: ISIC code for the product
        improvement_percentage: Percentage improvement (positive) or deterioration (negative).
                               For example, 10.0 means 10% improvement, -5.0 means 5% deterioration
        improvement_type: Type of improvement - options:
                         - "total_cost": affects total cost (inputs + value added)
                         - "intermediates": affects only production inputs
                         - "va": affects all value added components
                         - "specific_va": affects only the specified VA component (requires target_va_type)
        target_va_type: Specific value added type to improve when improvement_type="specific_va"
                       Options: "wages", "surplus", "taxes", "mixed_income"
        investment_cost: Optional specific investment cost, if not provided uses random value
    
    Returns:
        Investment: Configured investment object
    """
    # Fetch existing indice for the target ISIC to get current values
    gidb = GoodsIndiceDatabase()
    all_indices = gidb.get_all_good_indices()
    existing_indice = None
    for idx in all_indices:
        if str(idx.isic) == str(produce_isic):
            existing_indice = idx
            break
    
    if existing_indice is None:
        raise ValueError(f"No good indice found for ISIC: {produce_isic}")
    
    value_added_types = ["wages", "surplus", "taxes", "mixed_income"]
    
    # Get current values from the existing indice (convert from Column values to regular Python types)
    current_inputs = existing_indice.production_inputs if existing_indice.production_inputs is not None else {}
    current_added_values = existing_indice.production_added_values if existing_indice.production_added_values is not None else {}
    
    # Apply improvements/deteriorations based on type
    inputs = {}
    added_values = {}
    for k, v in current_inputs.items():
        inputs[k] = v
    for k, v in current_added_values.items():
        added_values[k] = v
    
    # Calculate improvement factor
    # Positive improvement_percentage means cost reduction (multiply by factor < 1.0)
    # Negative improvement_percentage means cost increase (multiply by factor > 1.0)
    # Example: 10% improvement -> factor = 0.9 (reduce by 10%)
    #         -10% deterioration -> factor = 1.1 (increase by 10%)
    improvement_factor = 1.0 - (improvement_percentage / 100.0)
    
    if improvement_type == "total_cost":
        # Apply improvement to both inputs and value added
        inputs = {isic: max(1, int(float(value) * improvement_factor)) for isic, value in inputs.items()}
        added_values = {va_type: max(1, int(float(value) * improvement_factor)) for va_type, value in added_values.items()}
    
    elif improvement_type == "intermediates":
        # Apply improvement only to production inputs
        inputs = {isic: max(1, int(float(value) * improvement_factor)) for isic, value in inputs.items()}
    
    elif improvement_type == "va":
        # Apply improvement to all value added components
        added_values = {va_type: max(1, int(float(value) * improvement_factor)) for va_type, value in added_values.items()}
    
    elif improvement_type == "specific_va":
        # Apply improvement to specific value added component
        if target_va_type and target_va_type in added_values:
            added_values[target_va_type] = max(1, int(float(added_values[target_va_type]) * improvement_factor))
        else:
            raise ValueError(f"target_va_type must be specified and valid when improvement_type='specific_va'. Options: {value_added_types}")
    
    else:
        raise ValueError(f"Invalid improvement_type: {improvement_type}. Options: 'total_cost', 'intermediates', 'va', 'specific_va'")
    
    # Set investment cost
    if investment_cost is None:
        investment_cost = fake.random_number(digits=5)
    
    # Create investment
    investment = Investment(
        id=fake.random_int(min=1, max=1000), 
        name=f"{fake.word()}_{improvement_type}_{improvement_percentage}%", 
        type_of_investment=InvestmentType.INDICE, 
        produce_isic=produce_isic, 
        implementation_cost=investment_cost
    )
    
    investment.set_investment_metrics(inputs, added_values)
    
    return investment

def setup_random_sample_data():
    print("Testing Producer:")
    test_producer(10)
    print("\nTesting Good:")
    test_good(5,4)
    print("\nTesting Production inputs:")
    test_production_inputs()
    print("\nTesting completed.")
    print("\nEvaluating ISIC codes:")
    evaluate_goods_isic()
    print("Evaluation of ISIC codes completed.\nEvaluating Production Prices:")
    evaluate_productions_price()
    print("\nEvaluation of Production prices completed.\nEvaluating Indices Inputs and Prices:")
    evaluate_indicies()
    print("Evaluation of Indices Inputs and Prices completed.\nEvaluating Indices Production Inputs to Matrix:")
    evaluate_indicies_production_inputs_to_matrix()
    print("Evaluation of Indices Production Inputs to Matrix completed.")