import os
import json
import pandas as pd
import numpy as np
from pathlib import Path
from scratch.test_realistic_economy import create_realistic_economy

def generate_datasets():
    goods, prods = create_realistic_economy()
    
    # ISIC constants
    FX = 'A9999_999_999'
    AGR = 'A0111_100_01'
    ENE = 'A3510_200_02'
    IND = 'A2810_300_03'
    TRA = 'A4923_400_04'
    SRV = 'A6910_500_05'

    # Realistic Final Demand vector for 6 sectors (ordered alphabetically by ISIC):
    # AGR, IND, ENE, TRA, SRV, FX
    # Total FD = 2000.0 distributed realistically across consumer & capital sectors
    fd_str = "400.0;450.0;350.0;300.0;500.0;0.0"

    # Base tax policies CSV (Neutral Baseline, NO TAX CHANGES)
    # The user explicitly requested: "and we shall ignore tax changes"
    neutral_tax_policy = [{
        'example_id': 1,
        'change_type': 'tax_policy',
        'title': 'Realistic Economy Baseline - Stable Equilibrium',
        'description': 'Baseline national economic circular flow with realistic ISIC sectors|No tax policy changes (neutral 15% income tax, 25% corporate tax)|Dynamic circular flow with unallocated VA recycling',
        'final_demand': fd_str,
        'income_tax_rate_before': 0.15,
        'income_tax_rate_after': 0.15,
        'corporate_tax_rate_before': 0.25,
        'corporate_tax_rate_after': 0.25,
        'income_tax_applies_to': 'wages',
        'iterations': 5,
        'wage_proportions': '0.25;0.20;0.15;0.15;0.25;0.0',
        'surplus_proportions': '0.15;0.35;0.20;0.15;0.15;0.0',
        'government_proportions': '0.20;0.20;0.20;0.15;0.25;0.0',
        'tech_change_function_name': '',
        'tech_change_params': ''
    }]

    # National Scale Tech Changes (for ex3)
    national_tech_changes = [
        # Scenario 1: National Grid Decarbonization & Clean Power Efficiency
        {
            'example_id': 1,
            'tech_change_id': 'national_grid_efficiency',
            'change_type': 'tech_change',
            'title': 'National Clean Power Grid Efficiency (25% Reduction)',
            'description': 'Economy-wide energy efficiency standards reduce electric power inputs (A3510_200_02) by 25% across all production sectors.|Requires $300M grid modernization capital investment.',
            'final_demand': fd_str,
            'income_tax_rate_before': 0.15, 'income_tax_rate_after': 0.15,
            'corporate_tax_rate_before': 0.25, 'corporate_tax_rate_after': 0.25,
            'income_tax_applies_to': 'wages', 'iterations': 5,
            'wage_proportions': '', 'surplus_proportions': '', 'government_proportions': '',
            'use_multi_level': False, 'sequence_number': 1,
            'method': 'add_input_change', 'sector_idx': '', 'input_sector_idx': ENE,
            'production_id': '', 'isic': '', 'tier_index': '', 'field': '',
            'change_type_param': 'multiply', 'value': 0.75,
            'efficiency_type': '', 'va_component': '', 'input_isic': '', 'exclude_list': '',
            'position': '', 'cap': '', 'price': '', 'va_components': '',
            'requirements': '', 'investment_duration': ''
        },
        {
            'example_id': 1,
            'tech_change_id': 'national_grid_efficiency',
            'change_type': 'tech_change',
            'title': 'National Clean Power Grid Efficiency (25% Reduction)',
            'description': 'Economy-wide energy efficiency standards reduce electric power inputs (A3510_200_02) by 25% across all production sectors.|Requires $300M grid modernization capital investment.',
            'final_demand': fd_str,
            'income_tax_rate_before': 0.15, 'income_tax_rate_after': 0.15,
            'corporate_tax_rate_before': 0.25, 'corporate_tax_rate_after': 0.25,
            'income_tax_applies_to': 'wages', 'iterations': 5,
            'wage_proportions': '', 'surplus_proportions': '', 'government_proportions': '',
            'use_multi_level': False, 'sequence_number': 2,
            'method': 'set_capital_requirements', 'sector_idx': '', 'input_sector_idx': '',
            'production_id': '', 'isic': '', 'tier_index': '', 'field': '',
            'change_type_param': '', 'value': '',
            'efficiency_type': '', 'va_component': '', 'input_isic': '', 'exclude_list': '',
            'position': '', 'cap': '', 'price': '', 'va_components': '',
            'requirements': json.dumps({IND: 180.0, SRV: 120.0}), 'investment_duration': 2
        },

        # Scenario 2: National Intermodal Freight & Logistics Overhaul
        {
            'example_id': 2,
            'tech_change_id': 'national_logistics_modernization',
            'change_type': 'tech_change',
            'title': 'National Intermodal Freight & Logistics Overhaul (20% Input Reduction)',
            'description': 'High-efficiency rail corridors and digitized transport networks reduce the overall operational input requirements of the freight transport sector by 20%.',
            'final_demand': fd_str,
            'income_tax_rate_before': 0.15, 'income_tax_rate_after': 0.15,
            'corporate_tax_rate_before': 0.25, 'corporate_tax_rate_after': 0.25,
            'income_tax_applies_to': 'wages', 'iterations': 5,
            'wage_proportions': '', 'surplus_proportions': '', 'government_proportions': '',
            'use_multi_level': False, 'sequence_number': 1,
            'method': 'add_sector_change', 'sector_idx': TRA, 'input_sector_idx': '',
            'production_id': '', 'isic': '', 'tier_index': '', 'field': '',
            'change_type_param': 'multiply', 'value': 0.80,
            'efficiency_type': '', 'va_component': '', 'input_isic': '', 'exclude_list': '',
            'position': '', 'cap': '', 'price': '', 'va_components': '',
            'requirements': '', 'investment_duration': ''
        },
        {
            'example_id': 2,
            'tech_change_id': 'national_logistics_modernization',
            'change_type': 'tech_change',
            'title': 'National Intermodal Freight & Logistics Overhaul (20% Input Reduction)',
            'description': 'High-efficiency rail corridors and digitized transport networks reduce the overall operational input requirements of the freight transport sector by 20%.',
            'final_demand': fd_str,
            'income_tax_rate_before': 0.15, 'income_tax_rate_after': 0.15,
            'corporate_tax_rate_before': 0.25, 'corporate_tax_rate_after': 0.25,
            'income_tax_applies_to': 'wages', 'iterations': 5,
            'wage_proportions': '', 'surplus_proportions': '', 'government_proportions': '',
            'use_multi_level': False, 'sequence_number': 2,
            'method': 'set_capital_requirements', 'sector_idx': '', 'input_sector_idx': '',
            'production_id': '', 'isic': '', 'tier_index': '', 'field': '',
            'change_type_param': '', 'value': '',
            'efficiency_type': '', 'va_component': '', 'input_isic': '', 'exclude_list': '',
            'position': '', 'cap': '', 'price': '', 'va_components': '',
            'requirements': json.dumps({IND: 150.0, TRA: 100.0}), 'investment_duration': 2
        },

        # Scenario 3: Industrial Automation & IT Service Substitution
        {
            'example_id': 3,
            'tech_change_id': 'industrial_digital_automation',
            'change_type': 'tech_change',
            'title': 'Industrial Automation & IT Substitution',
            'description': 'Machinery and chemical plants automate manufacturing: reduces manual agricultural/raw inputs by 30% while expanding digital IT software inputs by 25%.',
            'final_demand': fd_str,
            'income_tax_rate_before': 0.15, 'income_tax_rate_after': 0.15,
            'corporate_tax_rate_before': 0.25, 'corporate_tax_rate_after': 0.25,
            'income_tax_applies_to': 'wages', 'iterations': 5,
            'wage_proportions': '', 'surplus_proportions': '', 'government_proportions': '',
            'use_multi_level': False, 'sequence_number': 1,
            'method': 'add_coefficient_change', 'sector_idx': IND, 'input_sector_idx': AGR,
            'production_id': '', 'isic': '', 'tier_index': '', 'field': '',
            'change_type_param': 'multiply', 'value': 0.70,
            'efficiency_type': '', 'va_component': '', 'input_isic': '', 'exclude_list': '',
            'position': '', 'cap': '', 'price': '', 'va_components': '',
            'requirements': '', 'investment_duration': ''
        },
        {
            'example_id': 3,
            'tech_change_id': 'industrial_digital_automation',
            'change_type': 'tech_change',
            'title': 'Industrial Automation & IT Substitution',
            'description': 'Machinery and chemical plants automate manufacturing: reduces manual agricultural/raw inputs by 30% while expanding digital IT software inputs by 25%.',
            'final_demand': fd_str,
            'income_tax_rate_before': 0.15, 'income_tax_rate_after': 0.15,
            'corporate_tax_rate_before': 0.25, 'corporate_tax_rate_after': 0.25,
            'income_tax_applies_to': 'wages', 'iterations': 5,
            'wage_proportions': '', 'surplus_proportions': '', 'government_proportions': '',
            'use_multi_level': False, 'sequence_number': 2,
            'method': 'add_coefficient_change', 'sector_idx': IND, 'input_sector_idx': SRV,
            'production_id': '', 'isic': '', 'tier_index': '', 'field': '',
            'change_type_param': 'multiply', 'value': 1.25,
            'efficiency_type': '', 'va_component': '', 'input_isic': '', 'exclude_list': '',
            'position': '', 'cap': '', 'price': '', 'va_components': '',
            'requirements': '', 'investment_duration': ''
        },
        {
            'example_id': 3,
            'tech_change_id': 'industrial_digital_automation',
            'change_type': 'tech_change',
            'title': 'Industrial Automation & IT Substitution',
            'description': 'Machinery and chemical plants automate manufacturing: reduces manual agricultural/raw inputs by 30% while expanding digital IT software inputs by 25%.',
            'final_demand': fd_str,
            'income_tax_rate_before': 0.15, 'income_tax_rate_after': 0.15,
            'corporate_tax_rate_before': 0.25, 'corporate_tax_rate_after': 0.25,
            'income_tax_applies_to': 'wages', 'iterations': 5,
            'wage_proportions': '', 'surplus_proportions': '', 'government_proportions': '',
            'use_multi_level': False, 'sequence_number': 3,
            'method': 'set_capital_requirements', 'sector_idx': '', 'input_sector_idx': '',
            'production_id': '', 'isic': '', 'tier_index': '', 'field': '',
            'change_type_param': '', 'value': '',
            'efficiency_type': '', 'va_component': '', 'input_isic': '', 'exclude_list': '',
            'position': '', 'cap': '', 'price': '', 'va_components': '',
            'requirements': json.dumps({IND: 100.0, SRV: 80.0}), 'investment_duration': 1
        },

        # Scenario 4: Strategic Machinery Manufacturing Capacity Expansion
        {
            'example_id': 4,
            'tech_change_id': 'machinery_capacity_expansion',
            'change_type': 'tech_change',
            'title': 'Strategic Domestic Machinery Capacity Expansion',
            'description': 'Doubles domestic automated machinery manufacturing capacity (Tier 0) from 85 to 170 units to replace costly foreign machinery imports.',
            'final_demand': fd_str,
            'income_tax_rate_before': 0.15, 'income_tax_rate_after': 0.15,
            'corporate_tax_rate_before': 0.25, 'corporate_tax_rate_after': 0.25,
            'income_tax_applies_to': 'wages', 'iterations': 5,
            'wage_proportions': '', 'surplus_proportions': '', 'government_proportions': '',
            'use_multi_level': True, 'sequence_number': 1,
            'method': 'add_curve_tier_change', 'sector_idx': IND, 'input_sector_idx': '',
            'production_id': '', 'isic': IND, 'tier_index': 0, 'field': 'cap',
            'change_type_param': 'multiply', 'value': 2.0,
            'efficiency_type': '', 'va_component': '', 'input_isic': '', 'exclude_list': '',
            'position': '', 'cap': '', 'price': '', 'va_components': '',
            'requirements': '', 'investment_duration': ''
        },
        {
            'example_id': 4,
            'tech_change_id': 'machinery_capacity_expansion',
            'change_type': 'tech_change',
            'title': 'Strategic Domestic Machinery Capacity Expansion',
            'description': 'Doubles domestic automated machinery manufacturing capacity (Tier 0) from 85 to 170 units to replace costly foreign machinery imports.',
            'final_demand': fd_str,
            'income_tax_rate_before': 0.15, 'income_tax_rate_after': 0.15,
            'corporate_tax_rate_before': 0.25, 'corporate_tax_rate_after': 0.25,
            'income_tax_applies_to': 'wages', 'iterations': 5,
            'wage_proportions': '', 'surplus_proportions': '', 'government_proportions': '',
            'use_multi_level': True, 'sequence_number': 2,
            'method': 'set_capital_requirements', 'sector_idx': '', 'input_sector_idx': '',
            'production_id': '', 'isic': IND, 'tier_index': 0, 'field': 'cap',
            'change_type_param': '', 'value': '',
            'efficiency_type': '', 'va_component': '', 'input_isic': '', 'exclude_list': '',
            'position': '', 'cap': '', 'price': '', 'va_components': '',
            'requirements': json.dumps({IND: 200.0}), 'investment_duration': 2
        },

        # Scenario 5: National Green Industry & Energy Transition Portfolio
        {
            'example_id': 5,
            'tech_change_id': 'national_green_transition',
            'change_type': 'tech_change',
            'title': 'National Green Industry & Energy Transition Portfolio',
            'description': 'Coordinated economy-wide transformation: 20% power efficiency across all sectors, 15% manufacturing productivity boost, and 15% freight logistics efficiency.',
            'final_demand': fd_str,
            'income_tax_rate_before': 0.15, 'income_tax_rate_after': 0.15,
            'corporate_tax_rate_before': 0.25, 'corporate_tax_rate_after': 0.25,
            'income_tax_applies_to': 'wages', 'iterations': 5,
            'wage_proportions': '', 'surplus_proportions': '', 'government_proportions': '',
            'use_multi_level': False, 'sequence_number': 1,
            'method': 'add_input_change', 'sector_idx': '', 'input_sector_idx': ENE,
            'production_id': '', 'isic': '', 'tier_index': '', 'field': '',
            'change_type_param': 'multiply', 'value': 0.80,
            'efficiency_type': '', 'va_component': '', 'input_isic': '', 'exclude_list': '',
            'position': '', 'cap': '', 'price': '', 'va_components': '',
            'requirements': '', 'investment_duration': ''
        },
        {
            'example_id': 5,
            'tech_change_id': 'national_green_transition',
            'change_type': 'tech_change',
            'title': 'National Green Industry & Energy Transition Portfolio',
            'description': 'Coordinated economy-wide transformation: 20% power efficiency across all sectors, 15% manufacturing productivity boost, and 15% freight logistics efficiency.',
            'final_demand': fd_str,
            'income_tax_rate_before': 0.15, 'income_tax_rate_after': 0.15,
            'corporate_tax_rate_before': 0.25, 'corporate_tax_rate_after': 0.25,
            'income_tax_applies_to': 'wages', 'iterations': 5,
            'wage_proportions': '', 'surplus_proportions': '', 'government_proportions': '',
            'use_multi_level': False, 'sequence_number': 2,
            'method': 'add_sector_change', 'sector_idx': IND, 'input_sector_idx': '',
            'production_id': '', 'isic': '', 'tier_index': '', 'field': '',
            'change_type_param': 'multiply', 'value': 0.85,
            'efficiency_type': '', 'va_component': '', 'input_isic': '', 'exclude_list': '',
            'position': '', 'cap': '', 'price': '', 'va_components': '',
            'requirements': '', 'investment_duration': ''
        },
        {
            'example_id': 5,
            'tech_change_id': 'national_green_transition',
            'change_type': 'tech_change',
            'title': 'National Green Industry & Energy Transition Portfolio',
            'description': 'Coordinated economy-wide transformation: 20% power efficiency across all sectors, 15% manufacturing productivity boost, and 15% freight logistics efficiency.',
            'final_demand': fd_str,
            'income_tax_rate_before': 0.15, 'income_tax_rate_after': 0.15,
            'corporate_tax_rate_before': 0.25, 'corporate_tax_rate_after': 0.25,
            'income_tax_applies_to': 'wages', 'iterations': 5,
            'wage_proportions': '', 'surplus_proportions': '', 'government_proportions': '',
            'use_multi_level': False, 'sequence_number': 3,
            'method': 'add_sector_change', 'sector_idx': TRA, 'input_sector_idx': '',
            'production_id': '', 'isic': '', 'tier_index': '', 'field': '',
            'change_type_param': 'multiply', 'value': 0.85,
            'efficiency_type': '', 'va_component': '', 'input_isic': '', 'exclude_list': '',
            'position': '', 'cap': '', 'price': '', 'va_components': '',
            'requirements': '', 'investment_duration': ''
        },
        {
            'example_id': 5,
            'tech_change_id': 'national_green_transition',
            'change_type': 'tech_change',
            'title': 'National Green Industry & Energy Transition Portfolio',
            'description': 'Coordinated economy-wide transformation: 20% power efficiency across all sectors, 15% manufacturing productivity boost, and 15% freight logistics efficiency.',
            'final_demand': fd_str,
            'income_tax_rate_before': 0.15, 'income_tax_rate_after': 0.15,
            'corporate_tax_rate_before': 0.25, 'corporate_tax_rate_after': 0.25,
            'income_tax_applies_to': 'wages', 'iterations': 5,
            'wage_proportions': '', 'surplus_proportions': '', 'government_proportions': '',
            'use_multi_level': False, 'sequence_number': 4,
            'method': 'set_capital_requirements', 'sector_idx': '', 'input_sector_idx': '',
            'production_id': '', 'isic': '', 'tier_index': '', 'field': '',
            'change_type_param': '', 'value': '',
            'efficiency_type': '', 'va_component': '', 'input_isic': '', 'exclude_list': '',
            'position': '', 'cap': '', 'price': '', 'va_components': '',
            'requirements': json.dumps({IND: 250.0, ENE: 150.0, TRA: 100.0}), 'investment_duration': 3
        }
    ]

    # Firm Scale Tech Changes (for ex4)
    # Target specific production facilities (prods 1 to 25)
    firm_tech_changes = [
        # Scenario 1: Farm-Level Precision Agritech Recipe Upgrade
        {
            'example_id': 1,
            'tech_change_id': 'farm_recipe_upgrade',
            'change_type': 'tech_change',
            'title': 'Farm-Level Precision Agritech Recipe Modernization',
            'description': 'Commercial mechanized farm (Prod ID 2) adopts smart soil sensors and drip irrigation: reduces power pumping costs from $18 to $11 and machinery parts from $24 to $16.',
            'final_demand': fd_str,
            'income_tax_rate_before': 0.15, 'income_tax_rate_after': 0.15,
            'corporate_tax_rate_before': 0.25, 'corporate_tax_rate_after': 0.25,
            'income_tax_applies_to': 'wages', 'iterations': 5,
            'wage_proportions': '', 'surplus_proportions': '', 'government_proportions': '',
            'use_multi_level': True, 'sequence_number': 1,
            'method': 'add_production_input_change', 'sector_idx': '', 'input_sector_idx': '',
            'production_id': 2, 'isic': '', 'tier_index': '', 'field': '',
            'change_type_param': 'replace_cost', 'value': 11.0,
            'efficiency_type': '', 'va_component': '', 'input_isic': ENE, 'exclude_list': '',
            'position': '', 'cap': '', 'price': '', 'va_components': '',
            'requirements': '', 'investment_duration': ''
        },
        {
            'example_id': 1,
            'tech_change_id': 'farm_recipe_upgrade',
            'change_type': 'tech_change',
            'title': 'Farm-Level Precision Agritech Recipe Modernization',
            'description': 'Commercial mechanized farm (Prod ID 2) adopts smart soil sensors and drip irrigation: reduces power pumping costs from $18 to $11 and machinery parts from $24 to $16.',
            'final_demand': fd_str,
            'income_tax_rate_before': 0.15, 'income_tax_rate_after': 0.15,
            'corporate_tax_rate_before': 0.25, 'corporate_tax_rate_after': 0.25,
            'income_tax_applies_to': 'wages', 'iterations': 5,
            'wage_proportions': '', 'surplus_proportions': '', 'government_proportions': '',
            'use_multi_level': True, 'sequence_number': 2,
            'method': 'add_production_input_change', 'sector_idx': '', 'input_sector_idx': '',
            'production_id': 2, 'isic': '', 'tier_index': '', 'field': '',
            'change_type_param': 'replace_cost', 'value': 16.0,
            'efficiency_type': '', 'va_component': '', 'input_isic': IND, 'exclude_list': '',
            'position': '', 'cap': '', 'price': '', 'va_components': '',
            'requirements': '', 'investment_duration': ''
        },
        {
            'example_id': 1,
            'tech_change_id': 'farm_recipe_upgrade',
            'change_type': 'tech_change',
            'title': 'Farm-Level Precision Agritech Recipe Modernization',
            'description': 'Commercial mechanized farm (Prod ID 2) adopts smart soil sensors and drip irrigation: reduces power pumping costs from $18 to $11 and machinery parts from $24 to $16.',
            'final_demand': fd_str,
            'income_tax_rate_before': 0.15, 'income_tax_rate_after': 0.15,
            'corporate_tax_rate_before': 0.25, 'corporate_tax_rate_after': 0.25,
            'income_tax_applies_to': 'wages', 'iterations': 5,
            'wage_proportions': '', 'surplus_proportions': '', 'government_proportions': '',
            'use_multi_level': True, 'sequence_number': 3,
            'method': 'add_coefficient_change', 'sector_idx': AGR, 'input_sector_idx': ENE,
            'production_id': '', 'isic': '', 'tier_index': '', 'field': '',
            'change_type_param': 'multiply', 'value': 0.85,
            'efficiency_type': '', 'va_component': '', 'input_isic': '', 'exclude_list': '',
            'position': '', 'cap': '', 'price': '', 'va_components': '',
            'requirements': '', 'investment_duration': ''
        },
        {
            'example_id': 1,
            'tech_change_id': 'farm_recipe_upgrade',
            'change_type': 'tech_change',
            'title': 'Farm-Level Precision Agritech Recipe Modernization',
            'description': 'Commercial mechanized farm (Prod ID 2) adopts smart soil sensors and drip irrigation: reduces power pumping costs from $18 to $11 and machinery parts from $24 to $16.',
            'final_demand': fd_str,
            'income_tax_rate_before': 0.15, 'income_tax_rate_after': 0.15,
            'corporate_tax_rate_before': 0.25, 'corporate_tax_rate_after': 0.25,
            'income_tax_applies_to': 'wages', 'iterations': 5,
            'wage_proportions': '', 'surplus_proportions': '', 'government_proportions': '',
            'use_multi_level': True, 'sequence_number': 4,
            'method': 'set_capital_requirements', 'sector_idx': '', 'input_sector_idx': '',
            'production_id': '', 'isic': '', 'tier_index': '', 'field': '',
            'change_type_param': '', 'value': '',
            'efficiency_type': '', 'va_component': '', 'input_isic': '', 'exclude_list': '',
            'position': '', 'cap': '', 'price': '', 'va_components': '',
            'requirements': json.dumps({IND: 60.0}), 'investment_duration': 1
        },

        # Scenario 2: Heavy Equipment Works Bio-Composite Material Substitution
        {
            'example_id': 2,
            'tech_change_id': 'firm_biocomposite_substitution',
            'change_type': 'tech_change',
            'title': 'Firm Recipe: Domestic Bio-Composite Material Substitution',
            'description': 'Heavy Equipment Plant (Prod ID 12) substitutes petroleum chemicals with domestic agricultural bio-composites: increases agricultural input from $20 to $32 while improving unit margins.',
            'final_demand': fd_str,
            'income_tax_rate_before': 0.15, 'income_tax_rate_after': 0.15,
            'corporate_tax_rate_before': 0.25, 'corporate_tax_rate_after': 0.25,
            'income_tax_applies_to': 'wages', 'iterations': 5,
            'wage_proportions': '', 'surplus_proportions': '', 'government_proportions': '',
            'use_multi_level': True, 'sequence_number': 1,
            'method': 'add_production_input_change', 'sector_idx': '', 'input_sector_idx': '',
            'production_id': 12, 'isic': '', 'tier_index': '', 'field': '',
            'change_type_param': 'replace_cost', 'value': 32.0,
            'efficiency_type': '', 'va_component': '', 'input_isic': AGR, 'exclude_list': '',
            'position': '', 'cap': '', 'price': '', 'va_components': '',
            'requirements': '', 'investment_duration': ''
        },
        {
            'example_id': 2,
            'tech_change_id': 'firm_biocomposite_substitution',
            'change_type': 'tech_change',
            'title': 'Firm Recipe: Domestic Bio-Composite Material Substitution',
            'description': 'Heavy Equipment Plant (Prod ID 12) substitutes petroleum chemicals with domestic agricultural bio-composites: increases agricultural input from $20 to $32 while improving unit margins.',
            'final_demand': fd_str,
            'income_tax_rate_before': 0.15, 'income_tax_rate_after': 0.15,
            'corporate_tax_rate_before': 0.25, 'corporate_tax_rate_after': 0.25,
            'income_tax_applies_to': 'wages', 'iterations': 5,
            'wage_proportions': '', 'surplus_proportions': '', 'government_proportions': '',
            'use_multi_level': True, 'sequence_number': 2,
            'method': 'add_coefficient_change', 'sector_idx': IND, 'input_sector_idx': AGR,
            'production_id': '', 'isic': '', 'tier_index': '', 'field': '',
            'change_type_param': 'multiply', 'value': 1.20,
            'efficiency_type': '', 'va_component': '', 'input_isic': '', 'exclude_list': '',
            'position': '', 'cap': '', 'price': '', 'va_components': '',
            'requirements': '', 'investment_duration': ''
        },
        {
            'example_id': 2,
            'tech_change_id': 'firm_biocomposite_substitution',
            'change_type': 'tech_change',
            'title': 'Firm Recipe: Domestic Bio-Composite Material Substitution',
            'description': 'Heavy Equipment Plant (Prod ID 12) substitutes petroleum chemicals with domestic agricultural bio-composites: increases agricultural input from $20 to $32 while improving unit margins.',
            'final_demand': fd_str,
            'income_tax_rate_before': 0.15, 'income_tax_rate_after': 0.15,
            'corporate_tax_rate_before': 0.25, 'corporate_tax_rate_after': 0.25,
            'income_tax_applies_to': 'wages', 'iterations': 5,
            'wage_proportions': '', 'surplus_proportions': '', 'government_proportions': '',
            'use_multi_level': True, 'sequence_number': 3,
            'method': 'set_capital_requirements', 'sector_idx': '', 'input_sector_idx': '',
            'production_id': '', 'isic': '', 'tier_index': '', 'field': '',
            'change_type_param': '', 'value': '',
            'efficiency_type': '', 'va_component': '', 'input_isic': '', 'exclude_list': '',
            'position': '', 'cap': '', 'price': '', 'va_components': '',
            'requirements': json.dumps({IND: 75.0}), 'investment_duration': 1
        },

        # Scenario 3: Plant Automation & Labor Efficiency Overhaul
        {
            'example_id': 3,
            'tech_change_id': 'firm_plant_automation',
            'change_type': 'tech_change',
            'title': 'Plant Automation: Assembly Line Lean Modernization',
            'description': 'Machinery & Valve Fabricator (Prod ID 13) installs automated robotics: reduces all material and energy inputs by 15% and improves labor efficiency.',
            'final_demand': fd_str,
            'income_tax_rate_before': 0.15, 'income_tax_rate_after': 0.15,
            'corporate_tax_rate_before': 0.25, 'corporate_tax_rate_after': 0.25,
            'income_tax_applies_to': 'wages', 'iterations': 5,
            'wage_proportions': '', 'surplus_proportions': '', 'government_proportions': '',
            'use_multi_level': True, 'sequence_number': 1,
            'method': 'add_production_all_inputs_change', 'sector_idx': '', 'input_sector_idx': '',
            'production_id': 13, 'isic': '', 'tier_index': '', 'field': '',
            'change_type_param': 'multiply', 'value': 0.85,
            'efficiency_type': '', 'va_component': '', 'input_isic': '', 'exclude_list': '',
            'position': '', 'cap': '', 'price': '', 'va_components': '',
            'requirements': '', 'investment_duration': ''
        },
        {
            'example_id': 3,
            'tech_change_id': 'firm_plant_automation',
            'change_type': 'tech_change',
            'title': 'Plant Automation: Assembly Line Lean Modernization',
            'description': 'Machinery & Valve Fabricator (Prod ID 13) installs automated robotics: reduces all material and energy inputs by 15% and improves labor efficiency.',
            'final_demand': fd_str,
            'income_tax_rate_before': 0.15, 'income_tax_rate_after': 0.15,
            'corporate_tax_rate_before': 0.25, 'corporate_tax_rate_after': 0.25,
            'income_tax_applies_to': 'wages', 'iterations': 5,
            'wage_proportions': '', 'surplus_proportions': '', 'government_proportions': '',
            'use_multi_level': True, 'sequence_number': 2,
            'method': 'add_production_efficiency_change', 'sector_idx': '', 'input_sector_idx': '',
            'production_id': 13, 'isic': '', 'tier_index': '', 'field': '',
            'change_type_param': 'add', 'value': 20.0,
            'efficiency_type': 'material', 'va_component': '', 'input_isic': '', 'exclude_list': '',
            'position': '', 'cap': '', 'price': '', 'va_components': '',
            'requirements': '', 'investment_duration': ''
        },
        {
            'example_id': 3,
            'tech_change_id': 'firm_plant_automation',
            'change_type': 'tech_change',
            'title': 'Plant Automation: Assembly Line Lean Modernization',
            'description': 'Machinery & Valve Fabricator (Prod ID 13) installs automated robotics: reduces all material and energy inputs by 15% and improves labor efficiency.',
            'final_demand': fd_str,
            'income_tax_rate_before': 0.15, 'income_tax_rate_after': 0.15,
            'corporate_tax_rate_before': 0.25, 'corporate_tax_rate_after': 0.25,
            'income_tax_applies_to': 'wages', 'iterations': 5,
            'wage_proportions': '', 'surplus_proportions': '', 'government_proportions': '',
            'use_multi_level': True, 'sequence_number': 3,
            'method': 'set_capital_requirements', 'sector_idx': '', 'input_sector_idx': '',
            'production_id': '', 'isic': '', 'tier_index': '', 'field': '',
            'change_type_param': '', 'value': '',
            'efficiency_type': '', 'va_component': '', 'input_isic': '', 'exclude_list': '',
            'position': '', 'cap': '', 'price': '', 'va_components': '',
            'requirements': json.dumps({IND: 80.0, SRV: 20.0}), 'investment_duration': 2
        },

        # Scenario 4: High-Efficiency Multimodal Freight Hub Expansion
        {
            'example_id': 4,
            'tech_change_id': 'freight_hub_expansion',
            'change_type': 'tech_change',
            'title': 'Firm Scale: Multimodal Freight Terminal Capacity Doubling',
            'description': 'Multimodal Freight Hub (Prod ID 16) doubles its low-cost electric rail and green dispatch terminal capacity from 90 to 180 units.',
            'final_demand': fd_str,
            'income_tax_rate_before': 0.15, 'income_tax_rate_after': 0.15,
            'corporate_tax_rate_before': 0.25, 'corporate_tax_rate_after': 0.25,
            'income_tax_applies_to': 'wages', 'iterations': 5,
            'wage_proportions': '', 'surplus_proportions': '', 'government_proportions': '',
            'use_multi_level': True, 'sequence_number': 1,
            'method': 'add_curve_tier_change', 'sector_idx': TRA, 'input_sector_idx': '',
            'production_id': '', 'isic': TRA, 'tier_index': 0, 'field': 'cap',
            'change_type_param': 'multiply', 'value': 2.0,
            'efficiency_type': '', 'va_component': '', 'input_isic': '', 'exclude_list': '',
            'position': '', 'cap': '', 'price': '', 'va_components': '',
            'requirements': '', 'investment_duration': ''
        },
        {
            'example_id': 4,
            'tech_change_id': 'freight_hub_expansion',
            'change_type': 'tech_change',
            'title': 'Firm Scale: Multimodal Freight Terminal Capacity Doubling',
            'description': 'Multimodal Freight Hub (Prod ID 16) doubles its low-cost electric rail and green dispatch terminal capacity from 90 to 180 units.',
            'final_demand': fd_str,
            'income_tax_rate_before': 0.15, 'income_tax_rate_after': 0.15,
            'corporate_tax_rate_before': 0.25, 'corporate_tax_rate_after': 0.25,
            'income_tax_applies_to': 'wages', 'iterations': 5,
            'wage_proportions': '', 'surplus_proportions': '', 'government_proportions': '',
            'use_multi_level': True, 'sequence_number': 2,
            'method': 'set_capital_requirements', 'sector_idx': '', 'input_sector_idx': '',
            'production_id': '', 'isic': TRA, 'tier_index': 0, 'field': 'cap',
            'change_type_param': '', 'value': '',
            'efficiency_type': '', 'va_component': '', 'input_isic': '', 'exclude_list': '',
            'position': '', 'cap': '', 'price': '', 'va_components': '',
            'requirements': json.dumps({TRA: 60.0, IND: 40.0}), 'investment_duration': 1
        },

        # Scenario 5: Gas Turbine Power Plant Modernization
        {
            'example_id': 5,
            'tech_change_id': 'turbine_retrofit_upgrade',
            'change_type': 'tech_change',
            'title': 'Firm Scale: Gas Turbine Heat-Recovery & Automation Retrofit',
            'description': 'Combined-Cycle Station (Prod ID 7) retrofits turbine with digital control: reduces industrial maintenance parts input by 25% and cuts transport fuel delivery cost by 30%.',
            'final_demand': fd_str,
            'income_tax_rate_before': 0.15, 'income_tax_rate_after': 0.15,
            'corporate_tax_rate_before': 0.25, 'corporate_tax_rate_after': 0.25,
            'income_tax_applies_to': 'wages', 'iterations': 5,
            'wage_proportions': '', 'surplus_proportions': '', 'government_proportions': '',
            'use_multi_level': True, 'sequence_number': 1,
            'method': 'add_production_input_change', 'sector_idx': '', 'input_sector_idx': '',
            'production_id': 7, 'isic': '', 'tier_index': '', 'field': '',
            'change_type_param': 'multiply', 'value': 0.75,
            'efficiency_type': '', 'va_component': '', 'input_isic': IND, 'exclude_list': '',
            'position': '', 'cap': '', 'price': '', 'va_components': '',
            'requirements': '', 'investment_duration': ''
        },
        {
            'example_id': 5,
            'tech_change_id': 'turbine_retrofit_upgrade',
            'change_type': 'tech_change',
            'title': 'Firm Scale: Gas Turbine Heat-Recovery & Automation Retrofit',
            'description': 'Combined-Cycle Station (Prod ID 7) retrofits turbine with digital control: reduces industrial maintenance parts input by 25% and cuts transport fuel delivery cost by 30%.',
            'final_demand': fd_str,
            'income_tax_rate_before': 0.15, 'income_tax_rate_after': 0.15,
            'corporate_tax_rate_before': 0.25, 'corporate_tax_rate_after': 0.25,
            'income_tax_applies_to': 'wages', 'iterations': 5,
            'wage_proportions': '', 'surplus_proportions': '', 'government_proportions': '',
            'use_multi_level': True, 'sequence_number': 2,
            'method': 'add_production_input_change', 'sector_idx': '', 'input_sector_idx': '',
            'production_id': 7, 'isic': '', 'tier_index': '', 'field': '',
            'change_type_param': 'multiply', 'value': 0.70,
            'efficiency_type': '', 'va_component': '', 'input_isic': TRA, 'exclude_list': '',
            'position': '', 'cap': '', 'price': '', 'va_components': '',
            'requirements': '', 'investment_duration': ''
        },
        {
            'example_id': 5,
            'tech_change_id': 'turbine_retrofit_upgrade',
            'change_type': 'tech_change',
            'title': 'Firm Scale: Gas Turbine Heat-Recovery & Automation Retrofit',
            'description': 'Combined-Cycle Station (Prod ID 7) retrofits turbine with digital control: reduces industrial maintenance parts input by 25% and cuts transport fuel delivery cost by 30%.',
            'final_demand': fd_str,
            'income_tax_rate_before': 0.15, 'income_tax_rate_after': 0.15,
            'corporate_tax_rate_before': 0.25, 'corporate_tax_rate_after': 0.25,
            'income_tax_applies_to': 'wages', 'iterations': 5,
            'wage_proportions': '', 'surplus_proportions': '', 'government_proportions': '',
            'use_multi_level': True, 'sequence_number': 3,
            'method': 'add_coefficient_change', 'sector_idx': ENE, 'input_sector_idx': IND,
            'production_id': '', 'isic': '', 'tier_index': '', 'field': '',
            'change_type_param': 'multiply', 'value': 0.88,
            'efficiency_type': '', 'va_component': '', 'input_isic': '', 'exclude_list': '',
            'position': '', 'cap': '', 'price': '', 'va_components': '',
            'requirements': '', 'investment_duration': ''
        },
        {
            'example_id': 5,
            'tech_change_id': 'turbine_retrofit_upgrade',
            'change_type': 'tech_change',
            'title': 'Firm Scale: Gas Turbine Heat-Recovery & Automation Retrofit',
            'description': 'Combined-Cycle Station (Prod ID 7) retrofits turbine with digital control: reduces industrial maintenance parts input by 25% and cuts transport fuel delivery cost by 30%.',
            'final_demand': fd_str,
            'income_tax_rate_before': 0.15, 'income_tax_rate_after': 0.15,
            'corporate_tax_rate_before': 0.25, 'corporate_tax_rate_after': 0.25,
            'income_tax_applies_to': 'wages', 'iterations': 5,
            'wage_proportions': '', 'surplus_proportions': '', 'government_proportions': '',
            'use_multi_level': True, 'sequence_number': 4,
            'method': 'set_capital_requirements', 'sector_idx': '', 'input_sector_idx': '',
            'production_id': '', 'isic': '', 'tier_index': '', 'field': '',
            'change_type_param': '', 'value': '',
            'efficiency_type': '', 'va_component': '', 'input_isic': '', 'exclude_list': '',
            'position': '', 'cap': '', 'price': '', 'va_components': '',
            'requirements': json.dumps({IND: 110.0}), 'investment_duration': 2
        }
    ]

    # For ex3, we can also include the Firm Scale scenarios as Scenarios 6 to 10!
    all_tech_changes = list(national_tech_changes)
    for row in firm_tech_changes:
        r_copy = dict(row)
        r_copy['example_id'] = row['example_id'] + 5 # 6 to 10
        all_tech_changes.append(r_copy)

    # Function to save dataset
    def save_dataset(folder_name, tc_list):
        out_dir = Path("data") / folder_name
        out_dir.mkdir(parents=True, exist_ok=True)
        
        # 1. goods.csv
        pd.DataFrame(goods).to_csv(out_dir / "goods.csv", index=False)
        
        # 2. productions.csv
        expected_prod_cols = ['id','name','descriptive_name','id_number','isic','producer','produce','produce_name','production_inputs','production_added_values','production_rate','production_quantity','price']
        dummy_cols = ['production_material_efficiency', 'production_labour_efficiency', 'production_energy_efficiency', 'contact_name', 'contact_email', 'contact_phone', 'contact_phone2', 'contact_website', 'address', 'address_street', 'address_city', 'address_country', 'address_postal_code', 'total_inputs_cost', 'total_value_added']
        prod_df = pd.DataFrame(prods)
        for d in dummy_cols:
            prod_df[d] = ''
        full_cols = expected_prod_cols[:12] + dummy_cols + ['price']
        prod_df[full_cols].to_csv(out_dir / "productions.csv", index=False)
        
        # 3. tax_policies.csv (neutral baseline)
        pd.DataFrame(neutral_tax_policy).to_csv(out_dir / "tax_policies.csv", index=False)
        
        # 4. tech_changes.csv
        tc_cols = ['example_id','tech_change_id','change_type','title','description','final_demand','income_tax_rate_before','income_tax_rate_after','corporate_tax_rate_before','corporate_tax_rate_after','income_tax_applies_to','iterations','wage_proportions','surplus_proportions','government_proportions','use_multi_level','sequence_number','method','sector_idx','input_sector_idx','production_id','isic','tier_index','field','change_type_param','value','efficiency_type','va_component','input_isic','exclude_list','position','cap','price','va_components','requirements','investment_duration']
        tc_df = pd.DataFrame(tc_list)
        for c in tc_cols:
            if c not in tc_df.columns:
                tc_df[c] = ''
        tc_df[tc_cols].to_csv(out_dir / "tech_changes.csv", index=False)
        print(f"Saved dataset '{folder_name}' with {len(goods)} goods, {len(prods)} productions, and {len(tc_list)} tech change rows.")

    # Save ex3 (Full Realistic Suite: Examples 1-5 National Scale, Examples 6-10 Firm Scale)
    save_dataset("ex3", all_tech_changes)

    # Save ex4 (Dedicated Realistic Firm Scale: Examples 1-5 Firm Scale)
    save_dataset("ex4", firm_tech_changes)

if __name__ == '__main__':
    generate_datasets()
