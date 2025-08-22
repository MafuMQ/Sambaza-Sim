import logging
from Producer import ProducersDatabase
from Good import GoodsDatabase
from Good_Indice import GoodsIndiceDatabase
from Production import ProductionsDatabase

logging.basicConfig(level=logging.INFO)

def test_producer():
    pdb = ProducersDatabase()
    # Add producer 1
    producer1 = pdb.add_producer(name="Test Producer 1", descriptive_name="A test producer", id_number=1001)
    print("Added producer 1:", producer1)
    # Add producer 2
    producer2 = pdb.add_producer(name="Test Producer 2", descriptive_name="Another test producer", id_number=1002)
    print("Added producer 2:", producer2)
    # Get by id
    fetched = pdb.get_producer_by_id(producer1.id)
    print("Fetched by id:", fetched)
    # Update
    updated = pdb.update_producer(producer1.id, descriptive_name="Updated producer")
    print("Updated producer:", updated)
    # Search
    found = pdb.search_producers("Test")
    print("Search result:", found)
    # Delete only producer 1
    deleted = pdb.delete_producer(producer1.id)
    print("Deleted producer 1:", deleted)

def test_good():
    gdb = GoodsDatabase()
    # Add good 1
    good1 = gdb.add_good(name="Test Good 1", descriptive_name="A test good", id_number=2001, isic="A01")
    print("Added good 1:", good1)
    # Add good 2
    good2 = gdb.add_good(name="Test Good 2", descriptive_name="Another test good", id_number=2002, isic="A02")
    print("Added good 2:", good2)
    # Get all
    all_goods = gdb.get_all_goods()
    print("All goods:", all_goods)
    # Update
    updated = gdb.update_good(good1.id, descriptive_name="Updated good")
    print("Updated good:", updated)
    # Search
    found = gdb.search_goods("Test")
    print("Search result:", found)
    # Delete only good 1
    deleted = gdb.delete_good(good1.id)
    print("Deleted good 1:", deleted)

def test_good_indice():
    gidb = GoodsIndiceDatabase()
    # Add good indice 1
    indice1 = gidb.add_good_indice(name="Test Indice 1", id_number=3001, isic=101, price=50, quantity=100)
    print("Added good indice 1:", indice1)
    # Add good indice 2
    indice2 = gidb.add_good_indice(name="Test Indice 2", id_number=3002, isic=102, price=60, quantity=200)
    print("Added good indice 2:", indice2)
    # Get all
    all_indices = gidb.get_all_good_indices()
    print("All good indices:", all_indices)
    # Update
    updated = gidb.update_good_indice(3001, price=60)
    print("Updated good indice 1:", updated)
    # Delete only indice 1
    deleted = gidb.delete_good_indice(3001)
    print("Deleted good indice 1:", deleted)

def test_production():
    pdb = ProductionsDatabase()
    # Add production 1
    production1 = pdb.add_production(name="Test Production 1", id_number=4001, producer=1001, produce=2001, produce_name="Test Good 1")
    print("Added production 1:", production1)
    # Add production 2
    production2 = pdb.add_production(name="Test Production 2", id_number=4002, producer=1002, produce=2002, produce_name="Test Good 2")
    print("Added production 2:", production2)
    # Get all
    all_productions = pdb.get_all_productions()
    print("All productions:", all_productions)
    # Update
    updated = pdb.update_production(production1.id, produce_name="Updated Production 1")
    print("Updated production 1:", updated)
    # Delete only production 1
    deleted = pdb.delete_production(production1.id)
    print("Deleted production 1:", deleted)

if __name__ == "__main__":
    print("Testing Producer:")
    test_producer()
    print("\nTesting Good:")
    test_good()
    print("\nTesting Good Indice:")
    test_good_indice()
    print("\nTesting Production:")
    test_production()