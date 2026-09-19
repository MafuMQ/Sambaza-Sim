import json
import numpy as np
import pandas as pd
from pathlib import Path

def create_realistic_economy():
    """Builds a realistic 6-sector economy (5 domestic sectors + Foreign Exchange)."""
    
    # 1. GOODS
    goods = [
        {
            'id': 1,
            'name': 'Foreign Exchange',
            'descriptive_name': 'Represents foreign currency used for international imports and global supply chains.',
            'id_number': 9999,
            'isic': 'A9999_999_999',
            'isic_section': 'A',
            'isic_division': '99',
            'isic_group': '9',
            'isic_class': '9',
            'sub_class_a': '9',
            'sub_class_b': '9',
            'sub_class_c': '9',
            'sub_class_nf': '999'
        },
        {
            'id': 2,
            'name': 'Agriculture & Food Products',
            'descriptive_name': 'Crops, livestock, grain farming, and commercial agro-processing.',
            'id_number': 1010,
            'isic': 'A0111_100_01',
            'isic_section': 'A',
            'isic_division': '01',
            'isic_group': '1',
            'isic_class': '1',
            'sub_class_a': '1',
            'sub_class_b': '0',
            'sub_class_c': '0',
            'sub_class_nf': '01'
        },
        {
            'id': 3,
            'name': 'Electric Power & Utilities',
            'descriptive_name': 'Power generation, clean renewable energy, transmission grid, and water utilities.',
            'id_number': 3510,
            'isic': 'A3510_200_02',
            'isic_section': 'A',
            'isic_division': '35',
            'isic_group': '1',
            'isic_class': '0',
            'sub_class_a': '2',
            'sub_class_b': '0',
            'sub_class_c': '0',
            'sub_class_nf': '02'
        },
        {
            'id': 4,
            'name': 'Industrial Machinery & Chemicals',
            'descriptive_name': 'Capital equipment, fabricated metals, chemicals, and advanced industrial machinery.',
            'id_number': 2810,
            'isic': 'A2810_300_03',
            'isic_section': 'A',
            'isic_division': '28',
            'isic_group': '1',
            'isic_class': '0',
            'sub_class_a': '3',
            'sub_class_b': '0',
            'sub_class_c': '0',
            'sub_class_nf': '03'
        },
        {
            'id': 5,
            'name': 'Transport & Freight Logistics',
            'descriptive_name': 'Freight transport by road, rail, inter-modal logistics, and warehousing services.',
            'id_number': 4923,
            'isic': 'A4923_400_04',
            'isic_section': 'A',
            'isic_division': '49',
            'isic_group': '2',
            'isic_class': '3',
            'sub_class_a': '4',
            'sub_class_b': '0',
            'sub_class_c': '0',
            'sub_class_nf': '04'
        },
        {
            'id': 6,
            'name': 'Commercial & IT Services',
            'descriptive_name': 'Business consulting, software, financial engineering, legal, and technical services.',
            'id_number': 6910,
            'isic': 'A6910_500_05',
            'isic_section': 'A',
            'isic_division': '69',
            'isic_group': '1',
            'isic_class': '0',
            'sub_class_a': '5',
            'sub_class_b': '0',
            'sub_class_c': '0',
            'sub_class_nf': '05'
        }
    ]

    FX = 'A9999_999_999'
    AGR = 'A0111_100_01'
    ENE = 'A3510_200_02'
    IND = 'A2810_300_03'
    TRA = 'A4923_400_04'
    SRV = 'A6910_500_05'

    # 2. PRODUCTIONS
    # Each domestic good has 4 domestic tiers + 1 IMPORT tier (total 5 productions per good = 25 productions)
    # Sorted in merit order (cheapest tier first, higher cost later, import last)
    prods = []
    pid = 1

    # --- Good 2: Agriculture & Food Products (id_number: 1010) ---
    # Tier 1 (Cheapest): Modern Precision Agritech Farm
    # inputs: Energy 12, Machinery 18, Transport 10, Services 8 = 48; VA: wages 22, surplus 15 = 37 -> price 85
    prods.append({
        'id': pid, 'name': 'Precision Agritech Mega-Farm', 'descriptive_name': 'Drip-irrigated GPS-guided precision agriculture with automated harvesting.',
        'id_number': 4101, 'isic': AGR, 'producer': 1001, 'produce': 1010, 'produce_name': 'Agriculture & Food Products',
        'production_inputs': json.dumps({ENE: 12.0, IND: 18.0, TRA: 10.0, SRV: 8.0}),
        'production_added_values': json.dumps({'wages': 22.0, 'surplus': 15.0}),
        'production_rate': 65, 'production_quantity': 80, 'price': 85.0
    }); pid += 1

    # Tier 2: Mechanized Commercial Agro-Hub
    # inputs: Energy 18, Machinery 24, Transport 14, Services 10 = 66; VA: wages 26, surplus 18 = 44 -> price 110
    prods.append({
        'id': pid, 'name': 'Commercial Mechanized Farm Cooperative', 'descriptive_name': 'Standard mechanized grain and food production facility.',
        'id_number': 4102, 'isic': AGR, 'producer': 1002, 'produce': 1010, 'produce_name': 'Agriculture & Food Products',
        'production_inputs': json.dumps({ENE: 18.0, IND: 24.0, TRA: 14.0, SRV: 10.0}),
        'production_added_values': json.dumps({'wages': 26.0, 'surplus': 18.0}),
        'production_rate': 55, 'production_quantity': 95, 'price': 110.0
    }); pid += 1

    # Tier 3: Traditional Regional Cooperative
    # inputs: Energy 22, Machinery 30, Transport 18, Services 10 = 80; VA: wages 32, surplus 18 = 50 -> price 130
    prods.append({
        'id': pid, 'name': 'Regional Farming Cooperative', 'descriptive_name': 'Semi-mechanized traditional farming network.',
        'id_number': 4103, 'isic': AGR, 'producer': 1003, 'produce': 1010, 'produce_name': 'Agriculture & Food Products',
        'production_inputs': json.dumps({ENE: 22.0, IND: 30.0, TRA: 18.0, SRV: 10.0}),
        'production_added_values': json.dumps({'wages': 32.0, 'surplus': 18.0}),
        'production_rate': 45, 'production_quantity': 75, 'price': 130.0
    }); pid += 1

    # Tier 4: Smallholder Intensive Farming
    # inputs: Energy 28, Machinery 36, Transport 24, Services 12 = 100; VA: wages 36, surplus 16 = 52 -> price 152
    prods.append({
        'id': pid, 'name': 'Smallholder Independent Growers', 'descriptive_name': 'Labor-intensive smallholder crop production.',
        'id_number': 4104, 'isic': AGR, 'producer': 1004, 'produce': 1010, 'produce_name': 'Agriculture & Food Products',
        'production_inputs': json.dumps({ENE: 28.0, IND: 36.0, TRA: 24.0, SRV: 12.0}),
        'production_added_values': json.dumps({'wages': 36.0, 'surplus': 16.0}),
        'production_rate': 30, 'production_quantity': 60, 'price': 152.0
    }); pid += 1

    # Tier 5: IMPORT
    prods.append({
        'id': pid, 'name': 'IMPORT', 'descriptive_name': 'Imported grain, bulk livestock feed, and packaged international food items.',
        'id_number': 91010, 'isic': AGR, 'producer': 99999, 'produce': 1010, 'produce_name': 'Agriculture & Food Products',
        'production_inputs': json.dumps({FX: 260.0}),
        'production_added_values': json.dumps({'tariffs': 20.0}),
        'production_rate': 0, 'production_quantity': -1, 'price': 280.0
    }); pid += 1


    # --- Good 3: Electric Power & Utilities (id_number: 3510) ---
    # Tier 1: Modern Renewable Combined-Cycle Complex
    # inputs: Machinery 16, Transport 8, Services 12 = 36; VA: wages 24, surplus 20 = 44 -> price 80
    prods.append({
        'id': pid, 'name': 'Clean Energy Hydro & Solar Complex', 'descriptive_name': 'High-efficiency renewable energy generation plant and solar farm.',
        'id_number': 4201, 'isic': ENE, 'producer': 2001, 'produce': 3510, 'produce_name': 'Electric Power & Utilities',
        'production_inputs': json.dumps({IND: 16.0, TRA: 8.0, SRV: 12.0}),
        'production_added_values': json.dumps({'wages': 24.0, 'surplus': 20.0}),
        'production_rate': 75, 'production_quantity': 110, 'price': 80.0
    }); pid += 1

    # Tier 2: Combined-Cycle Gas & Grid Utility
    # inputs: Industrial 26, Transport 14, Services 16 = 56; VA: wages 28, surplus 24 = 52 -> price 108
    prods.append({
        'id': pid, 'name': 'Combined-Cycle Gas Turbine Station', 'descriptive_name': 'Modern natural gas and grid distribution utility.',
        'id_number': 4202, 'isic': ENE, 'producer': 2002, 'produce': 3510, 'produce_name': 'Electric Power & Utilities',
        'production_inputs': json.dumps({IND: 26.0, TRA: 14.0, SRV: 16.0}),
        'production_added_values': json.dumps({'wages': 28.0, 'surplus': 24.0}),
        'production_rate': 60, 'production_quantity': 90, 'price': 108.0
    }); pid += 1

    # Tier 3: Municipal Baseload Power Plant
    # inputs: Industrial 38, Transport 22, Services 18 = 78; VA: wages 32, surplus 22 = 54 -> price 132
    prods.append({
        'id': pid, 'name': 'Municipal Baseload Power Plant', 'descriptive_name': 'Conventional baseload thermal power plant.',
        'id_number': 4203, 'isic': ENE, 'producer': 2003, 'produce': 3510, 'produce_name': 'Electric Power & Utilities',
        'production_inputs': json.dumps({IND: 38.0, TRA: 22.0, SRV: 18.0}),
        'production_added_values': json.dumps({'wages': 32.0, 'surplus': 22.0}),
        'production_rate': 45, 'production_quantity': 80, 'price': 132.0
    }); pid += 1

    # Tier 4: Peaker Reserve Generation Plant
    # inputs: Industrial 52, Transport 30, Services 24 = 106; VA: wages 34, surplus 20 = 54 -> price 160
    prods.append({
        'id': pid, 'name': 'Emergency Peaker Generation Station', 'descriptive_name': 'Quick-start reserve generation for peak load handling.',
        'id_number': 4204, 'isic': ENE, 'producer': 2004, 'produce': 3510, 'produce_name': 'Electric Power & Utilities',
        'production_inputs': json.dumps({IND: 52.0, TRA: 30.0, SRV: 24.0}),
        'production_added_values': json.dumps({'wages': 34.0, 'surplus': 20.0}),
        'production_rate': 25, 'production_quantity': 50, 'price': 160.0
    }); pid += 1

    # Tier 5: IMPORT
    prods.append({
        'id': pid, 'name': 'IMPORT', 'descriptive_name': 'Cross-border interconnection electric power and liquefied natural gas imports.',
        'id_number': 93510, 'isic': ENE, 'producer': 99999, 'produce': 3510, 'produce_name': 'Electric Power & Utilities',
        'production_inputs': json.dumps({FX: 280.0}),
        'production_added_values': json.dumps({'tariffs': 25.0}),
        'production_rate': 0, 'production_quantity': -1, 'price': 305.0
    }); pid += 1


    # --- Good 4: Industrial Machinery & Chemicals (id_number: 2810) ---
    # Tier 1: Smart Automated Manufacturing Plant
    # inputs: Agriculture/Materials 14, Energy 18, Transport 12, Services 16 = 60; VA: wages 32, surplus 28 = 60 -> price 120
    prods.append({
        'id': pid, 'name': 'Automated Smart Machine Works', 'descriptive_name': 'Advanced robotic tooling, CNC precision engineering and chemical synthesis.',
        'id_number': 4301, 'isic': IND, 'producer': 3001, 'produce': 2810, 'produce_name': 'Industrial Machinery & Chemicals',
        'production_inputs': json.dumps({AGR: 14.0, ENE: 18.0, TRA: 12.0, SRV: 16.0}),
        'production_added_values': json.dumps({'wages': 32.0, 'surplus': 28.0}),
        'production_rate': 70, 'production_quantity': 85, 'price': 120.0
    }); pid += 1

    # Tier 2: Heavy Engineering & Equipment Fabrication
    # inputs: Agriculture/Materials 20, Energy 26, Transport 18, Services 22 = 86; VA: wages 36, surplus 26 = 62 -> price 148
    prods.append({
        'id': pid, 'name': 'Heavy Equipment & Machinery Works', 'descriptive_name': 'Industrial machinery, boilers, presses, and chemical reactor vessels.',
        'id_number': 4302, 'isic': IND, 'producer': 3002, 'produce': 2810, 'produce_name': 'Industrial Machinery & Chemicals',
        'production_inputs': json.dumps({AGR: 20.0, ENE: 26.0, TRA: 18.0, SRV: 22.0}),
        'production_added_values': json.dumps({'wages': 36.0, 'surplus': 26.0}),
        'production_rate': 55, 'production_quantity': 75, 'price': 148.0
    }); pid += 1

    # Tier 3: Standard Industrial Tool & Valve Factory
    # inputs: Agriculture/Materials 26, Energy 36, Transport 24, Services 28 = 114; VA: wages 40, surplus 24 = 64 -> price 178
    prods.append({
        'id': pid, 'name': 'Standard Machinery & Valve Fabricator', 'descriptive_name': 'Standardized industrial equipment and replacement components.',
        'id_number': 4303, 'isic': IND, 'producer': 3003, 'produce': 2810, 'produce_name': 'Industrial Machinery & Chemicals',
        'production_inputs': json.dumps({AGR: 26.0, ENE: 36.0, TRA: 24.0, SRV: 28.0}),
        'production_added_values': json.dumps({'wages': 40.0, 'surplus': 24.0}),
        'production_rate': 40, 'production_quantity': 65, 'price': 178.0
    }); pid += 1

    # Tier 4: Legacy Foundry & Batch Workshop
    # inputs: Agriculture/Materials 34, Energy 48, Transport 32, Services 34 = 148; VA: wages 44, surplus 22 = 66 -> price 214
    prods.append({
        'id': pid, 'name': 'Legacy Foundry & Machining Works', 'descriptive_name': 'Manual casting foundry and labor-intensive mechanical repair works.',
        'id_number': 4304, 'isic': IND, 'producer': 3004, 'produce': 2810, 'produce_name': 'Industrial Machinery & Chemicals',
        'production_inputs': json.dumps({AGR: 34.0, ENE: 48.0, TRA: 32.0, SRV: 34.0}),
        'production_added_values': json.dumps({'wages': 44.0, 'surplus': 22.0}),
        'production_rate': 25, 'production_quantity': 45, 'price': 214.0
    }); pid += 1

    # Tier 5: IMPORT
    prods.append({
        'id': pid, 'name': 'IMPORT', 'descriptive_name': 'Imported high-precision capital equipment, semiconductors, and specialized chemicals.',
        'id_number': 92810, 'isic': IND, 'producer': 99999, 'produce': 2810, 'produce_name': 'Industrial Machinery & Chemicals',
        'production_inputs': json.dumps({FX: 380.0}),
        'production_added_values': json.dumps({'tariffs': 35.0}),
        'production_rate': 0, 'production_quantity': -1, 'price': 415.0
    }); pid += 1


    # --- Good 5: Transport & Freight Logistics (id_number: 4923) ---
    # Tier 1: Green Fleet & Multimodal Rail Logistics
    # inputs: Energy 14, Machinery 12, Services 10 = 36; VA: wages 26, surplus 18 = 44 -> price 80
    prods.append({
        'id': pid, 'name': 'Multimodal Freight & Green Hub', 'descriptive_name': 'Electric rail, green fleet haulage, and automated cold-chain logistics.',
        'id_number': 4401, 'isic': TRA, 'producer': 4001, 'produce': 4923, 'produce_name': 'Transport & Freight Logistics',
        'production_inputs': json.dumps({ENE: 14.0, IND: 12.0, SRV: 10.0}),
        'production_added_values': json.dumps({'wages': 26.0, 'surplus': 18.0}),
        'production_rate': 70, 'production_quantity': 90, 'price': 80.0
    }); pid += 1

    # Tier 2: Commercial Highway Haulage Carrier
    # inputs: Energy 22, Machinery 18, Services 16 = 56; VA: wages 30, surplus 20 = 50 -> price 106
    prods.append({
        'id': pid, 'name': 'Commercial Interstate Freight Carrier', 'descriptive_name': 'Scheduled highway freight fleet and container trucking.',
        'id_number': 4402, 'isic': TRA, 'producer': 4002, 'produce': 4923, 'produce_name': 'Transport & Freight Logistics',
        'production_inputs': json.dumps({ENE: 22.0, IND: 18.0, SRV: 16.0}),
        'production_added_values': json.dumps({'wages': 30.0, 'surplus': 20.0}),
        'production_rate': 55, 'production_quantity': 80, 'price': 106.0
    }); pid += 1

    # Tier 3: Regional Distribution & Warehousing
    # inputs: Energy 30, Machinery 24, Services 22 = 76; VA: wages 34, surplus 20 = 54 -> price 130
    prods.append({
        'id': pid, 'name': 'Regional Trucking & Warehousing Hub', 'descriptive_name': 'Regional distribution centers and mid-size fleet transport.',
        'id_number': 4403, 'isic': TRA, 'producer': 4003, 'produce': 4923, 'produce_name': 'Transport & Freight Logistics',
        'production_inputs': json.dumps({ENE: 30.0, IND: 24.0, SRV: 22.0}),
        'production_added_values': json.dumps({'wages': 34.0, 'surplus': 20.0}),
        'production_rate': 40, 'production_quantity': 70, 'price': 130.0
    }); pid += 1

    # Tier 4: Independent Operator Fleet
    # inputs: Energy 40, Machinery 32, Services 28 = 100; VA: wages 38, surplus 18 = 56 -> price 156
    prods.append({
        'id': pid, 'name': 'Independent Owner-Operator Network', 'descriptive_name': 'Independent owner-operator carriers and spot freight services.',
        'id_number': 4404, 'isic': TRA, 'producer': 4004, 'produce': 4923, 'produce_name': 'Transport & Freight Logistics',
        'production_inputs': json.dumps({ENE: 40.0, IND: 32.0, SRV: 28.0}),
        'production_added_values': json.dumps({'wages': 38.0, 'surplus': 18.0}),
        'production_rate': 30, 'production_quantity': 50, 'price': 156.0
    }); pid += 1

    # Tier 5: IMPORT
    prods.append({
        'id': pid, 'name': 'IMPORT', 'descriptive_name': 'International ocean container shipping, air express cargo, and foreign logistics.',
        'id_number': 94923, 'isic': TRA, 'producer': 99999, 'produce': 4923, 'produce_name': 'Transport & Freight Logistics',
        'production_inputs': json.dumps({FX: 290.0}),
        'production_added_values': json.dumps({'tariffs': 25.0}),
        'production_rate': 0, 'production_quantity': -1, 'price': 315.0
    }); pid += 1


    # --- Good 6: Commercial & IT Services (id_number: 6910) ---
    # Tier 1: Digital Cloud & Automation Consultancy
    # inputs: Energy 8, Machinery 10, Transport 6 = 24; VA: wages 46, surplus 30 = 76 -> price 100
    prods.append({
        'id': pid, 'name': 'Digital Cloud & AI Solutions Bureau', 'descriptive_name': 'Enterprise cloud software, system integration, and automated financial services.',
        'id_number': 4501, 'isic': SRV, 'producer': 5001, 'produce': 6910, 'produce_name': 'Commercial & IT Services',
        'production_inputs': json.dumps({ENE: 8.0, IND: 10.0, TRA: 6.0}),
        'production_added_values': json.dumps({'wages': 46.0, 'surplus': 30.0}),
        'production_rate': 80, 'production_quantity': 100, 'price': 100.0
    }); pid += 1

    # Tier 2: Corporate Professional & Legal Services
    # inputs: Energy 12, Machinery 16, Transport 10 = 38; VA: wages 54, surplus 34 = 88 -> price 126
    prods.append({
        'id': pid, 'name': 'Corporate Advisory & Legal Practice', 'descriptive_name': 'Audit, legal counsel, management consulting, and commercial brokerage.',
        'id_number': 4502, 'isic': SRV, 'producer': 5002, 'produce': 6910, 'produce_name': 'Commercial & IT Services',
        'production_inputs': json.dumps({ENE: 12.0, IND: 16.0, TRA: 10.0}),
        'production_added_values': json.dumps({'wages': 54.0, 'surplus': 34.0}),
        'production_rate': 60, 'production_quantity': 85, 'price': 126.0
    }); pid += 1

    # Tier 3: Technical Engineering & Testing Services
    # inputs: Energy 16, Machinery 24, Transport 14 = 54; VA: wages 62, surplus 36 = 98 -> price 152
    prods.append({
        'id': pid, 'name': 'Technical Engineering Design Bureau', 'descriptive_name': 'Civil engineering, environmental compliance, and architectural design.',
        'id_number': 4503, 'isic': SRV, 'producer': 5003, 'produce': 6910, 'produce_name': 'Commercial & IT Services',
        'production_inputs': json.dumps({ENE: 16.0, IND: 24.0, TRA: 14.0}),
        'production_added_values': json.dumps({'wages': 62.0, 'surplus': 36.0}),
        'production_rate': 45, 'production_quantity': 70, 'price': 152.0
    }); pid += 1

    # Tier 4: Regional Commercial Services Agency
    # inputs: Energy 22, Machinery 32, Transport 18 = 72; VA: wages 68, surplus 38 = 106 -> price 178
    prods.append({
        'id': pid, 'name': 'Regional Commercial Services Agency', 'descriptive_name': 'General business support, bookkeeping, and local market research services.',
        'id_number': 4504, 'isic': SRV, 'producer': 5004, 'produce': 6910, 'produce_name': 'Commercial & IT Services',
        'production_inputs': json.dumps({ENE: 22.0, IND: 32.0, TRA: 18.0}),
        'production_added_values': json.dumps({'wages': 68.0, 'surplus': 38.0}),
        'production_rate': 35, 'production_quantity': 55, 'price': 178.0
    }); pid += 1

    # Tier 5: IMPORT
    prods.append({
        'id': pid, 'name': 'IMPORT', 'descriptive_name': 'Global IT licensing, international reinsurance, and offshore consulting.',
        'id_number': 96910, 'isic': SRV, 'producer': 99999, 'produce': 6910, 'produce_name': 'Commercial & IT Services',
        'production_inputs': json.dumps({FX: 320.0}),
        'production_added_values': json.dumps({'tariffs': 28.0}),
        'production_rate': 0, 'production_quantity': -1, 'price': 348.0
    }); pid += 1

    return goods, prods

if __name__ == '__main__':
    goods, prods = create_realistic_economy()
    print(f"Generated {len(goods)} goods and {len(prods)} productions.")
    for p in prods:
        inputs = json.loads(p['production_inputs'])
        vas = json.loads(p['production_added_values'])
        tot = sum(inputs.values()) + sum(vas.values())
        assert abs(tot - p['price']) < 1e-4, f"Price mismatch in prod {p['name']}: {tot} vs {p['price']}"
    print("All production price constraints strictly verified!")
