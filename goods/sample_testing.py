import logging
from Producer import ProducersDatabase
from Good import GoodsDatabase
from Good_Indice import GoodsIndiceDatabase
from Production import ProductionsDatabase
import random
import json
from Evaluators import *
from growth.growth import create_leontief_inverse

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
        production_added_value = fake.random_int(min=1, max=1000)
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
            production_inputs=production_inputs, production_added_value=production_added_value,
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
    productions = pdb.get_all_productions()
    for production in productions:
        if production.name != "IMPORT":  # pyright: ignore[reportGeneralTypeIssues] # Skip IMPORT productions 
            inputs = {random.choice(existing_goods_isic): fake.random_int(min=1, max=100) for _ in range(3)}
            pdb.update_production(int(production.id), production_inputs=inputs) # pyright: ignore[reportArgumentType]

def test_setup():
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

if __name__ == "__main__":
    # test_setup()
    print("Evaluation of Indices Inputs and Prices completed.\nEvaluating Indices Production Inputs to Matrix:")
    A_matrix = evaluate_indicies_production_inputs_to_matrix()
    print("Evaluation of Indices Production Inputs to Matrix completed.")
    print(create_leontief_inverse(A_matrix))

#NOTE: every good has an import producer

#NOTE -1 issue -- this issue comes from e.g. A2244_121_482 = 1 * A9999_999_999 will auto resolve to 0 = 1 * A9999_999_999 - A2244_121_482

#ADVANCED: invest for.. employment & for productivity