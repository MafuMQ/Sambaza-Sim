import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

import logging
from Input_Output_Model.demos.util.Setup_Data import setup_data
from Input_Output_Model.demos.util.tax_policy_simulation import run_tax_policy_simulation

loggingLevel = logging.WARNING
logging.basicConfig(level=loggingLevel)
logger = logging.getLogger(__name__)

"""
Tax Policy Simulation Demos
============================

This module demonstrates various scenarios for Input-Output economic modeling with
circular flow dynamics and tax policy analysis.

Key Features:
- Value Added (VA) = Final Demand (FD) identity is maintained through proper scaling
- Circular flow: VA from period t becomes C+I+G in period t+1
- Tax policy comparison: analyze impact of tax rate changes on economy
- Multiple demand specification methods: direct, uniform, proportional, sector-specific
- Separate proportion vectors for Consumption, Investment, and Government spending
"""

# Example configurations
EXAMPLES = {
    1: {
        "title": "Circular Flow Model - Stable GDP (5 iterations)",
        "description": [
            "Demonstrates that with correct VA calculation, GDP remains stable across iterations",
            "in a closed economy with no savings. VA = FD in every period.",
            "Uses proportional distribution based on initial demand pattern."
        ],
        "params": {
            "total_demand": 1000.0,
            "proportions": [0.25, 0.20, 0.30, 0.15, 0.10, 0.0],
            "consumption_proportions": [0.25, 0.20, 0.30, 0.15, 0.10, 0.0],
            "investment_proportions": [0.25, 0.20, 0.30, 0.15, 0.10, 0.0],
            "government_proportions": [0.25, 0.20, 0.30, 0.15, 0.10, 0.0],
            "income_tax_rate_before": 0.10,
            "income_tax_rate_after": 0.10,
            "corporate_tax_rate_before": 0.25,
            "corporate_tax_rate_after": 0.25,
            "income_tax_applies_to": "bonusWages",
            "iterations": 5
        }
    },
    2: {
        "title": "Tax Policy Impact - Income Tax Increase (10% → 20%)",
        "description": [
            "Compare economy before and after income tax increase on bonus wages.",
            "Uses separate proportions for C, I, and G to show how different FD components",
            "are distributed across sectors."
        ],
        "params": {
            "total_demand": 500.0,
            "proportions": [0.30, 0.25, 0.20, 0.15, 0.10, 0.0],
            "consumption_proportions": [0.35, 0.30, 0.20, 0.10, 0.05, 0.0],  # Consumer goods
            "investment_proportions": [0.10, 0.15, 0.40, 0.25, 0.10, 0.0],   # Capital goods
            "government_proportions": [0.20, 0.20, 0.20, 0.20, 0.20, 0.0],   # Balanced
            "income_tax_rate_before": 0.10,
            "income_tax_rate_after": 0.20,
            "corporate_tax_rate_before": 0.25,
            "corporate_tax_rate_after": 0.25,
            "income_tax_applies_to": "bonusWages",
            "iterations": 3
        }
    },
    3: {
        "title": "Tax Policy Impact - Corporate Tax Increase (25% → 35%)",
        "description": [
            "Analyze how corporate tax increase affects investment and GDP.",
            "Investment heavily concentrated in capital goods sectors [2,3]."
        ],
        "params": {
            "total_demand": 500.0,
            "proportions": [0.20, 0.20, 0.30, 0.20, 0.10, 0.0],
            "consumption_proportions": [0.30, 0.25, 0.20, 0.15, 0.10, 0.0],  # Consumer pattern
            "investment_proportions": [0.05, 0.10, 0.45, 0.30, 0.10, 0.0],   # Heavy on capital
            "government_proportions": [0.25, 0.20, 0.20, 0.20, 0.15, 0.0],   # Balanced
            "income_tax_rate_before": 0.15,
            "income_tax_rate_after": 0.15,
            "corporate_tax_rate_before": 0.25,
            "corporate_tax_rate_after": 0.35,
            "income_tax_applies_to": "bonusWages",
            "iterations": 3
        }
    },
    4: {
        "title": "Sector-Specific Demand Shock with Spillover Effects",
        "description": [
            "Analyze the impact of $200 demand increase in sector [2].",
            "Shows how initial shock spreads through economy via C, I, G channels."
        ],
        "params": {
            "target_isic": "A2327_978_13",
            "demand_shock": 200.0,
            "consumption_proportions": [0.35, 0.30, 0.20, 0.10, 0.05, 0.0],  # Consumer goods
            "investment_proportions": [0.10, 0.15, 0.40, 0.25, 0.10, 0.0],   # Capital goods
            "government_proportions": [0.20, 0.20, 0.20, 0.20, 0.20, 0.0],   # Balanced
            "income_tax_rate_before": 0.15,
            "income_tax_rate_after": 0.20,
            "corporate_tax_rate_before": 0.25,
            "corporate_tax_rate_after": 0.30,
            "income_tax_applies_to": "bonusWages",
            "iterations": 3
        }
    },
    5: {
        "title": "Uniform Demand Distribution Across All FD Components",
        "description": [
            "Apply equal $75 initial demand to each domestic sector.",
            "C, I, and G are all uniformly distributed (equal shares to all sectors)."
        ],
        "params": {
            "uniform_demand": 75.0,
            "consumption_proportions": [0.20, 0.20, 0.20, 0.20, 0.20, 0.0],  # Uniform
            "investment_proportions": [0.20, 0.20, 0.20, 0.20, 0.20, 0.0],   # Uniform
            "government_proportions": [0.20, 0.20, 0.20, 0.20, 0.20, 0.0],   # Uniform
            "income_tax_rate_before": 0.12,
            "income_tax_rate_after": 0.18,
            "corporate_tax_rate_before": 0.20,
            "corporate_tax_rate_after": 0.28,
            "income_tax_applies_to": "wages",
            "iterations": 3
        }
    },
    6: {
        "title": "Highly Differentiated C, I, G Distributions",
        "description": [
            "Specify exact initial demand, then show extreme differentiation:",
            "  C: Heavily in consumer sectors [0,1]",
            "  I: Concentrated in capital/manufacturing [2,3]",
            "  G: Focused on services and infrastructure [1,4]"
        ],
        "params": {
            "demand_vector": [150.0, 120.0, 180.0, 90.0, 60.0, 0.0],
            "consumption_proportions": [0.40, 0.35, 0.15, 0.05, 0.05, 0.0],  # Consumer focus
            "investment_proportions": [0.05, 0.05, 0.50, 0.35, 0.05, 0.0],   # Manufacturing
            "government_proportions": [0.10, 0.30, 0.10, 0.10, 0.40, 0.0],   # Services
            "income_tax_rate_before": 0.15,
            "income_tax_rate_after": 0.25,
            "corporate_tax_rate_before": 0.25,
            "corporate_tax_rate_after": 0.25,
            "income_tax_applies_to": "both",
            "iterations": 3
        }
    }
}


def run_demo(example_id, examples_config=EXAMPLES):
    """
    Run a specific demo example.
    
    Parameters:
    -----------
    example_id : int or str
        The example number (1-6) or name to run
    examples_config : dict
        Dictionary containing all example configurations
    """
    # Convert to int if string number provided
    if isinstance(example_id, str) and example_id.isdigit():
        example_id = int(example_id)
    
    # Get example config
    if example_id not in examples_config:
        available = ', '.join(map(str, examples_config.keys()))
        raise ValueError(f"Example {example_id} not found. Available examples: {available}")
    
    config = examples_config[example_id]
    
    # Print header
    print("\n" + "="*100)
    print(f"EXAMPLE {example_id}: {config['title']}")
    print("="*100)
    for line in config['description']:
        print(line)
    print()
    
    # Run simulation with configured parameters
    run_tax_policy_simulation(**config['params'])


if __name__ == "__main__":
    print("Setting up sample data...")
    setup_data(source="data/ex2", overwrite_existing_data=True, logging_level=loggingLevel)
    print("\nData setup complete.\n")
    
    # Run all examples
    # for example_num in EXAMPLES.keys():
    #     run_demo(example_num)

    # print("\n" + "="*100)
    # print("All examples completed!")
    # print("="*100)

    # Run a specific example
    example_number = 2
    run_demo(example_number)
    
    print("\n" + "="*100)
    print(f'Example {example_number} completed!')
    print("="*100)