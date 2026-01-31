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

Parameter Guide:
---------------

Demand Specification (choose ONE):
  - demand_vector: Direct specification of demand for each sector
  - target_isic + demand_shock: Demand for specific sector (others get zero)
  - demand_shock only: Demand for random sector (others get zero)
  - uniform_demand: Equal demand across all sectors
  - total_demand + proportions: Distribute total by proportions

Tax Parameters:
  - income_tax_rate_before/after: Tax rate on wages (0.0 to 1.0)
  - corporate_tax_rate_before/after: Tax rate on surplus (0.0 to 1.0)
  - income_tax_applies_to: "minWages", "bonusWages", "wages", "both"

Circular Flow Parameters:
  - iterations: Number of circular flow iterations (default=1)
  - consumption_rate: Share of after-tax income consumed (0.0 to 1.0, default=1.0)
  - consumption_distribution: "proportional" or "uniform"

Example Usage:
-------------
    run_tax_policy_simulation(
        total_demand=1000.0,
        proportions=[0.25, 0.20, 0.30, 0.15, 0.10, 0.0],
        income_tax_rate_before=0.10,
        income_tax_rate_after=0.15,
        corporate_tax_rate_before=0.25,
        corporate_tax_rate_after=0.30,
        income_tax_applies_to="bonusWages",
        iterations=5
    )
"""

if __name__ == "__main__":
    print("Setting up sample data...")
    setup_data(source="data/ex2", overwrite_existing_data=True, logging_level=loggingLevel)
    print("\nData setup complete.\n")
    
    # ==================================================================================
    # EXAMPLE 1: Circular Flow - Stable GDP in Closed Economy
    # ==================================================================================
    # print("\n" + "="*100)
    # print("EXAMPLE 1: Circular Flow Model - Stable GDP (5 iterations)")
    # print("="*100)
    # print("Demonstrates that with correct VA calculation, GDP remains stable across iterations")
    # print("in a closed economy with no savings. VA = FD in every period.\n")
    
    # run_tax_policy_simulation(
    #     total_demand=1000.0,
    #     proportions=[0.25, 0.20, 0.30, 0.15, 0.10, 0.0],
    #     income_tax_rate_before=0.10,
    #     income_tax_rate_after=0.10,  # Same rates = no policy change
    #     corporate_tax_rate_before=0.25,
    #     corporate_tax_rate_after=0.25,
    #     income_tax_applies_to="bonusWages",
    #     iterations=5,
    #     consumption_rate=1.0,
    #     consumption_distribution="proportional"
    # )
    
    # ==================================================================================
    # EXAMPLE 2: Tax Policy Impact - Income Tax Increase
    # ==================================================================================
    print("\n\n" + "="*100)
    print("EXAMPLE 2: Tax Policy Impact - Income Tax Increase (10% → 20%)")
    print("="*100)
    print("Compare economy before and after income tax increase on bonus wages.\n")
    
    run_tax_policy_simulation(
        total_demand=500.0,
        proportions=[0.30, 0.25, 0.20, 0.15, 0.10, 0.0],
        income_tax_rate_before=0.10,
        income_tax_rate_after=0.20,  # Double the income tax
        corporate_tax_rate_before=0.25,
        corporate_tax_rate_after=0.25,
        income_tax_applies_to="bonusWages",
        iterations=3,
        consumption_rate=1.0,
        consumption_distribution="proportional"
    )
    
    # ==================================================================================
    # EXAMPLE 3: Tax Policy Impact - Corporate Tax Increase
    # ==================================================================================
    # print("\n\n" + "="*100)
    # print("EXAMPLE 3: Tax Policy Impact - Corporate Tax Increase (25% → 35%)")
    # print("="*100)
    # print("Analyze how corporate tax increase affects investment and GDP.\n")
    
    # run_tax_policy_simulation(
    #     total_demand=500.0,
    #     proportions=[0.20, 0.20, 0.30, 0.20, 0.10, 0.0],
    #     income_tax_rate_before=0.15,
    #     income_tax_rate_after=0.15,
    #     corporate_tax_rate_before=0.25,
    #     corporate_tax_rate_after=0.35,  # Increase corporate tax
    #     income_tax_applies_to="bonusWages",
    #     iterations=3,
    #     consumption_rate=1.0,
    #     consumption_distribution="proportional"
    # )
    
    # ==================================================================================
    # EXAMPLE 4: Sector-Specific Demand Shock
    # ==================================================================================
    # print("\n\n" + "="*100)
    # print("EXAMPLE 4: Sector-Specific Demand Shock")
    # print("="*100)
    # print("Analyze the impact of $200 demand increase in a specific sector.\n")
    
    # run_tax_policy_simulation(
    #     target_isic="A2327_978_13",  # Target specific sector
    #     demand_shock=200.0,
    #     income_tax_rate_before=0.15,
    #     income_tax_rate_after=0.20,
    #     corporate_tax_rate_before=0.25,
    #     corporate_tax_rate_after=0.30,
    #     income_tax_applies_to="bonusWages"
    # )
    
    # ==================================================================================
    # EXAMPLE 5: Uniform Demand Across All Sectors
    # ==================================================================================
    # print("\n\n" + "="*100)
    # print("EXAMPLE 5: Uniform Demand Distribution")
    # print("="*100)
    # print("Apply equal $75 demand to each domestic sector.\n")
    
    # run_tax_policy_simulation(
    #     uniform_demand=75.0,
    #     income_tax_rate_before=0.12,
    #     income_tax_rate_after=0.18,
    #     corporate_tax_rate_before=0.20,
    #     corporate_tax_rate_after=0.28,
    #     income_tax_applies_to="wages",
    #     iterations=3,
    #     consumption_rate=1.0,
    #     consumption_distribution="uniform"
    # )
    
    # ==================================================================================
    # EXAMPLE 6: Direct Demand Vector Specification
    # ==================================================================================
    # print("\n\n" + "="*100)
    # print("EXAMPLE 6: Direct Demand Vector Specification")
    # print("="*100)
    # print("Specify exact demand for each sector with custom vector.\n")
    
    # demand_vector = [150.0, 120.0, 180.0, 90.0, 60.0, 0.0]
    # run_tax_policy_simulation(
    #     demand_vector=demand_vector,
    #     income_tax_rate_before=0.15,
    #     income_tax_rate_after=0.25,
    #     corporate_tax_rate_before=0.25,
    #     corporate_tax_rate_after=0.25,
    #     income_tax_applies_to="both",  # Tax both min and bonus wages
    #     iterations=2,
    #     consumption_rate=1.0,
    #     consumption_distribution="proportional"
    # )
    
    # print("\n" + "="*100)
    # print("All examples completed!")
    # print("="*100)
