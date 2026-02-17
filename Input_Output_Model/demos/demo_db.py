"""
Demo Runner with Database Support
===================================

This module extends demo.py to support loading examples from the database.

Usage:
    # Load examples from database
    from demo_db import run_demo_from_db, list_examples_from_db
    
    list_examples_from_db()
    run_demo_from_db(1)
"""

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

import logging
from Input_Output_Model.demos.configs.load_tech_changes import (
    rebuild_examples_dict_from_db,
    get_tech_change_by_id
)
from Input_Output_Model.demos.util.Setup_Data import setup_data

logger = logging.getLogger(__name__)


def list_examples_from_db(database_url: str = "sqlite:///data.db"):
    """
    List all available examples from the database.
    
    Parameters:
    -----------
    database_url : str
        SQLite database URL
    """
    from Input_Output_Model.models.entities.TechChange import TechChangeDatabase
    
    db = TechChangeDatabase(database_url=database_url)
    tech_changes = db.get_all_tech_changes()
    
    if not tech_changes:
        print("\n" + "="*100)
        print("NO EXAMPLES FOUND IN DATABASE")
        print("="*100)
        print("\nPlease run setup_data() to load examples from CSV files.")
        return
    
    print("\n" + "="*100)
    print("AVAILABLE EXAMPLES FROM DATABASE")
    print("="*100)
    
    # Group by type
    tax_policy = [tc for tc in tech_changes if tc.change_type == 'tax_policy']
    tech_change = [tc for tc in tech_changes if tc.change_type == 'tech_change']
    
    if tax_policy:
        print("\nTax Policy Examples:")
        print("-" * 100)
        for tc in tax_policy:
            print(f"  {tc.example_id}. {tc.title}")
    
    if tech_change:
        print("\nTechnological Change Examples:")
        print("-" * 100)
        for tc in tech_change:
            print(f"  {tc.example_id}. {tc.title}")
            if tc.use_multi_level:
                print(f"      (Multi-Level)")
    
    print("\n" + "="*100)


def run_demo_from_db(example_id: int, database_url: str = "sqlite:///data.db"):
    """
    Run a demo example loaded from the database.
    
    Parameters:
    -----------
    example_id : int
        The example number to run
    database_url : str
        SQLite database URL
    """
    from Input_Output_Model.demos.demo import run_demo
    
    # Rebuild examples dictionary from database
    examples = rebuild_examples_dict_from_db(database_url=database_url)
    
    if not examples:
        print("\n" + "="*100)
        print("NO EXAMPLES FOUND IN DATABASE")
        print("="*100)
        print("\nPlease run setup_data() to load examples from CSV files.")
        return
    
    if example_id not in examples:
        available = ', '.join(map(str, examples.keys()))
        print(f"\n✗ Example {example_id} not found in database.")
        print(f"Available examples: {available}")
        print("\nRun list_examples_from_db() to see all available examples.")
        return
    
    # Run the demo using the loaded config
    run_demo(example_id, examples_config=examples)


def run_all_demos_from_db(database_url: str = "sqlite:///data.db"):
    """
    Run all demo examples from the database.
    
    Parameters:
    -----------
    database_url : str
        SQLite database URL
    """
    from Input_Output_Model.demos.demo import run_demo
    
    # Rebuild examples dictionary from database
    examples = rebuild_examples_dict_from_db(database_url=database_url)
    
    if not examples:
        print("\n" + "="*100)
        print("NO EXAMPLES FOUND IN DATABASE")
        print("="*100)
        print("\nPlease run setup_data() to load examples from CSV files.")
        return
    
    print(f"\nRunning all {len(examples)} examples from database...\n")
    
    for example_id in sorted(examples.keys()):
        run_demo(example_id, examples_config=examples)
        print("\n\n")


if __name__ == "__main__":
    import sys
    
    logging.basicConfig(level=logging.WARNING)
    
    # Check if data needs to be setup
    from pathlib import Path
    
    db_file = Path("data.db")
    if not db_file.exists():
        print("Database not found. Setting up data...")
        setup_data(source="data/ex2", overwrite_existing_data=True, logging_level=logging.WARNING)
        print()
    
    # List examples
    list_examples_from_db()
    
    # Run specific example if provided
    if len(sys.argv) > 1:
        try:
            example_num = int(sys.argv[1])
            print(f"\nRunning example {example_num}...\n")
            run_demo_from_db(example_num)
        except ValueError:
            print(f"Invalid example number: {sys.argv[1]}")
    else:
        print("\nUsage: python demo_db.py [example_number]")
        print("  or   python demo_db.py      # to list all examples")
