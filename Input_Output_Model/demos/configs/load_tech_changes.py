"""
Database Loading Utilities for Tech Changes
============================================

This module provides utilities to load tech change configurations from CSV files
into the data.db SQLite database, similar to how goods and productions are loaded.

Usage:
    from load_tech_changes import load_tech_changes_from_csv
    load_tech_changes_from_csv()
"""

import csv
import logging
from pathlib import Path
from typing import Optional, Dict, Any
import json

from Input_Output_Model.models.entities.TechChange import TechChangeDatabase

logger = logging.getLogger(__name__)


def build_tech_change_from_spec(spec: Dict, isic_map: Dict) -> 'TechnologicalChange':
    """
    Build a TechnologicalChange object from a data specification.
    
    This function reconstructs a TechnologicalChange object from pure data
    (loaded from CSV/database) without requiring hardcoded Python functions.
    
    Parameters:
    -----------
    spec : dict
        Specification with structure:
        {
            "name": "...",
            "description": "...",
            "changes": [
                {"method": "add_input_change", "params": {...}},
                ...
            ]
        }
    isic_map : dict
        Mapping of ISIC codes to matrix indices
    
    Returns:
    --------
    TechnologicalChange object with the specified changes
    """
    from Input_Output_Model.demos.util.technological_change import TechnologicalChange
    
    tech_change = TechnologicalChange(
        name=spec.get("name", "Unnamed Change"),
        description=spec.get("description", "")
    )
    
    # Apply each change method
    for change in spec.get("changes", []):
        method_name = change.get("method")
        params = change.get("params", {})
        
        if not hasattr(tech_change, method_name):
            logger.warning(f"Unknown method '{method_name}' in tech change spec")
            continue
        
        method = getattr(tech_change, method_name)
        try:
            method(**params)
        except Exception as e:
            logger.error(f"Failed to apply {method_name} with params {params}: {e}")
    
    return tech_change


def _deserialize_value(value: str, field_type: str = 'auto') -> Any:
    """
    Convert a CSV string back to Python value.
    
    Parameters:
    -----------
    value : str
        String value from CSV
    field_type : str
        Expected type: 'list', 'dict', 'float', 'int', 'bool', 'auto'
    """
    if not value or value == "":
        return None
    
    if field_type == 'list':
        # Semicolon-separated -> list of floats
        try:
            return [float(v) for v in value.split(";")]
        except ValueError:
            return value.split(";")
    elif field_type == 'dict':
        # JSON string -> dict
        return json.loads(value)
    elif field_type == 'float':
        return float(value)
    elif field_type == 'int':
        return int(value)
    elif field_type == 'bool':
        return value.lower() in ('true', '1', 'yes')
    else:
        # Auto-detect
        if ";" in value:
            try:
                return [float(v) for v in value.split(";")]
            except ValueError:
                return value.split(";")
        elif value.startswith('{') or value.startswith('['):
            try:
                return json.loads(value)
            except json.JSONDecodeError:
                return value
        elif value.lower() in ('true', 'false'):
            return value.lower() == 'true'
        else:
            try:
                if '.' in value:
                    return float(value)
                else:
                    return int(value)
            except ValueError:
                return value


def load_tax_policy_from_csv(
    csv_path: str = "data/tax_policy_examples.csv",
    database_url: str = "sqlite:///data.db",
    clear_existing: bool = False
) -> int:
    """
    Load tax policy examples from CSV into the database.
    
    Parameters:
    -----------
    csv_path : str
        Path to tax policy CSV file
    database_url : str
        SQLite database URL
    clear_existing : bool
        If True, clear all existing tech changes before loading
    
    Returns:
    --------
    int : Number of records loaded
    """
    db = TechChangeDatabase(database_url=database_url)
    count = 0
    
    try:
        # Clear existing if requested
        if clear_existing:
            db.clear_all()
            logger.info("Cleared existing tech change records")
        
        csv_file = Path(csv_path)
        if not csv_file.exists():
            logger.error(f"CSV file not found: {csv_path}")
            return 0
        
        with open(csv_file, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            
            for row in reader:
                try:
                    example_id = int(row['example_id'])
                    title = row['title']
                    description = row.get('description', '').replace(' | ', '\n')
                    
                    # Prepare parameters
                    params = {
                        'example_id': example_id,
                        'change_type': 'tax_policy',
                        'title': title,
                        'description': description
                    }
                    
                    # Add all parameter fields
                    field_mappings = {
                        'total_demand': 'float',
                        'uniform_demand': 'float',
                        'demand_vector': 'list',
                        'proportions': 'list',
                        'target_isic': 'auto',
                        'demand_shock': 'float',
                        'consumption_proportions': 'list',
                        'investment_proportions': 'list',
                        'government_proportions': 'list',
                        'income_tax_rate_before': 'float',
                        'income_tax_rate_after': 'float',
                        'corporate_tax_rate_before': 'float',
                        'corporate_tax_rate_after': 'float',
                        'income_tax_applies_to': 'auto',
                        'consumption_rate': 'float',
                        'iterations': 'int'
                    }
                    
                    for field, field_type in field_mappings.items():
                        if field in row and row[field]:
                            params[field] = _deserialize_value(row[field], field_type)
                    
                    # Add to database
                    db.add_tech_change(**params)
                    count += 1
                    
                except Exception as e:
                    logger.error(f"Failed to load tax policy example {row.get('example_id', '?')}: {e}")
        
        logger.info(f"Loaded {count} tax policy examples from {csv_path}")
        return count
    
    except Exception as e:
        logger.error(f"Failed to load tax policy CSV: {e}")
        return 0


def load_tech_change_from_csv(
    csv_path: str = "data/tech_change_examples.csv",
    database_url: str = "sqlite:///data.db",
    clear_existing: bool = False
) -> int:
    """
    Load tech change examples from CSV into the database.
    
    Parameters:
    -----------
    csv_path : str
        Path to tech change CSV file
    database_url : str
        SQLite database URL
    clear_existing : bool
        If True, clear all existing tech changes before loading
    
    Returns:
    --------
    int : Number of records loaded
    """
    db = TechChangeDatabase(database_url=database_url)
    count = 0
    
    try:
        # Clear existing if requested (only if not already cleared by tax policy load)
        if clear_existing:
            db.clear_all()
            logger.info("Cleared existing tech change records")
        
        csv_file = Path(csv_path)
        if not csv_file.exists():
            logger.error(f"CSV file not found: {csv_path}")
            return 0
        
        with open(csv_file, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            
            for row in reader:
                try:
                    example_id = int(row['example_id'])
                    title = row['title']
                    description = row.get('description', '').replace(' | ', '\n')
                    
                    # Prepare parameters
                    params = {
                        'example_id': example_id,
                        'change_type': 'tech_change',
                        'title': title,
                        'description': description,
                        'is_tech_comparison': True
                    }
                    
                    # Add tech change specific fields
                    if row.get('tech_change_function_name'):
                        params['tech_change_function_name'] = row['tech_change_function_name']
                    
                    # Parse and store JSON specifications
                    if row.get('tech_change_params'):
                        try:
                            tech_change_params = json.loads(row['tech_change_params'])
                            params['tech_change_params'] = tech_change_params
                            logger.info(f"Parsed tech_change_params for example {example_id}: {len(tech_change_params)} changes")
                        except json.JSONDecodeError as je:
                            logger.error(f"Failed to parse tech_change_params JSON for example {example_id}: {je}")
                    else:
                        logger.warning(f"No tech_change_params found for example {example_id}")
                    
                    logger.info(f"Params keys before add: {list(params.keys())}")
                    
                    if row.get('final_demand'):
                        params['final_demand'] = _deserialize_value(row['final_demand'], 'list')
                    
                    if row.get('use_multi_level'):
                        params['use_multi_level'] = _deserialize_value(row['use_multi_level'], 'bool')
                    
                    if row.get('solver_type'):
                        params['solver_type'] = row['solver_type']
                    
                    # Add tax policy fields if present (for combined examples)
                    field_mappings = {
                        'income_tax_rate_before': 'float',
                        'income_tax_rate_after': 'float',
                        'corporate_tax_rate_before': 'float',
                        'corporate_tax_rate_after': 'float',
                        'income_tax_applies_to': 'auto',
                        'consumption_proportions': 'list',
                        'investment_proportions': 'list',
                        'government_proportions': 'list',
                        'iterations': 'int'
                    }
                    
                    for field, field_type in field_mappings.items():
                        if field in row and row[field]:
                            params[field] = _deserialize_value(row[field], field_type)
                    
                    # Add to database
                    db.add_tech_change(**params)
                    count += 1
                    
                except Exception as e:
                    logger.error(f"Failed to load tech change example {row.get('example_id', '?')}: {e}")
        
        logger.info(f"Loaded {count} tech change examples from {csv_path}")
        return count
    
    except Exception as e:
        logger.error(f"Failed to load tech change CSV: {e}")
        return 0


def load_tech_changes_from_csv(
    tax_policy_csv: str = "data/tax_policy_examples.csv",
    tech_change_csv: str = "data/tech_change_examples.csv",
    database_url: str = "sqlite:///data.db",
    clear_existing: bool = True
) -> int:
    """
    Load all tech changes from CSV files into the database.
    
    Parameters:
    -----------
    tax_policy_csv : str or None
        Path to tax policy CSV file (None to skip)
    tech_change_csv : str or None
        Path to tech change CSV file (None to skip)
    database_url : str
        SQLite database URL
    clear_existing : bool
        If True, clear all existing tech changes before loading
    
    Returns:
    --------
    int : Total number of records loaded
    """
    total = 0
    
    # Load tax policy examples
    if tax_policy_csv:
        count = load_tax_policy_from_csv(
            csv_path=tax_policy_csv,
            database_url=database_url,
            clear_existing=clear_existing
        )
        total += count
    
    # Load tech change examples (don't clear again if already cleared)
    if tech_change_csv:
        count = load_tech_change_from_csv(
            csv_path=tech_change_csv,
            database_url=database_url,
            clear_existing=False if tax_policy_csv else clear_existing
        )
        total += count
    
    logger.info(f"Total tech changes loaded: {total}")
    return total


def get_tech_change_by_id(example_id: int, database_url: str = "sqlite:///data.db") -> Optional[Dict[str, Any]]:
    """
    Retrieve a tech change configuration from the database.
    
    Parameters:
    -----------
    example_id : int
        Example ID to retrieve
    database_url : str
        SQLite database URL
    
    Returns:
    --------
    dict : Tech change configuration in demo.py compatible format, or None if not found
    """
    db = TechChangeDatabase(database_url=database_url)
    tech_change = db.get_tech_change_by_id(example_id)
    
    if tech_change:
        return tech_change.to_dict()
    return None


def rebuild_examples_dict_from_db(database_url: str = "sqlite:///data.db") -> Dict[int, Dict]:
    """
    Rebuild the EXAMPLES dictionary from database, compatible with demo.py.
    
    This function loads all tech changes from the database and reconstructs
    the config format expected by run_demo().
    
    All tech change specifications are read from the database as JSON data -
    no hardcoded lookup tables needed.
    
    Parameters:
    -----------
    database_url : str
        SQLite database URL
    
    Returns:
    --------
    dict : Dictionary mapping example_id -> config dict
    """
    db = TechChangeDatabase(database_url=database_url)
    tech_changes = db.get_all_tech_changes()
    
    examples = {}
    
    for tc in tech_changes:
        config = tc.to_dict()
        
        # If it's a tech change example, build from JSON specification in database
        if tc.change_type == 'tech_change' and tc.tech_change_params:
            try:
                # Parse the JSON spec from database
                if isinstance(tc.tech_change_params, str):
                    changes = json.loads(tc.tech_change_params)
                else:
                    changes = tc.tech_change_params
                
                logger.info(f"Building tech change for example {tc.example_id} with {len(changes)} changes")
                
                # Build the spec structure
                spec = {
                    "name": tc.title,
                    "description": config['description'][0] if config['description'] else tc.title,
                    "changes": changes
                }
                
                # Create a builder lambda that uses the spec
                builder = lambda isic_map, s=spec: build_tech_change_from_spec(s, isic_map)
                config['params']['tech_change_config'] = {
                    'name': tc.title,
                    'description': config['description'][0] if config['description'] else tc.title,
                    'tech_change_builder': builder
                }
                logger.info(f"Successfully built tech_change_config for example {tc.example_id}")
            except Exception as e:
                logger.error(f"Failed to build tech change for example {tc.example_id}: {e}")
        
        examples[tc.example_id] = config
    
    logger.info(f"Rebuilt {len(examples)} examples from database")
    return examples


if __name__ == "__main__":
    # Example usage: load from CSV to database
    logging.basicConfig(level=logging.INFO)
    
    print("Loading tech changes from CSV to database...")
    total = load_tech_changes_from_csv()
    
    if total > 0:
        print(f"\n✓ Successfully loaded {total} tech change configurations")
        
        # Test retrieval
        print("\nTesting database retrieval...")
        example_1 = get_tech_change_by_id(1)
        if example_1:
            print(f"  Example 1: {example_1['title']}")
        
        # Test rebuild
        print("\nRebuilding EXAMPLES dict from database...")
        examples = rebuild_examples_dict_from_db()
        print(f"  Successfully rebuilt {len(examples)} examples")
    else:
        print("\n✗ Failed to load tech changes")
