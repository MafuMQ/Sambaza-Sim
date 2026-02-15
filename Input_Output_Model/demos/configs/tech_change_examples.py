"""
Technological Change Example Configurations
============================================

Configuration dictionaries for technological change simulation examples.
Each example demonstrates different types of technological innovation and their economic impacts.

Key Principle: Hold final demand CONSTANT to isolate the pure effect of technological change.

Change Levels:
- Level 1 (Matrix): Direct A matrix coefficient changes (fast, no rebuild)
- Level 2 (Productions): Changes to production records (requires matrix rebuild)
- Level 3 (Curves): Changes to supply curve tiers (requires supply data rebuild)
"""

import logging
from Input_Output_Model.demos.util.technological_change import (
    TechnologicalChange,
    create_energy_efficiency_change,
    create_productivity_improvement,
    LEVEL_MATRIX,
    LEVEL_PRODUCTION,
    LEVEL_CURVE
)

logger = logging.getLogger(__name__)


# Helper functions for building tech changes
def _find_sector_by_keyword(isic_map, keyword, fallback=0):
    """Find a sector containing a keyword in its ISIC code."""
    for isic, idx in isic_map.items():
        if keyword.lower() in isic.lower():
            return idx
    logger.warning(f"Sector with keyword '{keyword}' not found, using fallback index {fallback}")
    return fallback


def _create_automation_substitution():
    """Create automation that substitutes capital for labor."""
    tech = TechnologicalChange(
        name="Automation in Sector 3",
        description="Replace labor with machinery/capital"
    )
    # Reduce labor-intensive input (sector 0)
    tech.add_coefficient_change(
        sector_idx=3,
        input_sector_idx=0,
        change_type="multiply",
        value=0.60  # 40% reduction
    )
    # Increase capital input (sector 2)
    tech.add_coefficient_change(
        sector_idx=3,
        input_sector_idx=2,
        change_type="multiply",
        value=1.15  # 15% increase
    )
    return tech


def _create_material_efficiency():
    """Create material efficiency improvement."""
    tech = TechnologicalChange(
        name="Material Efficiency (Sector 4)",
        description="Reduce material waste in production"
    )
    # Reduce material input from sector 1
    tech.add_coefficient_change(
        sector_idx=4,
        input_sector_idx=1,
        change_type="multiply",
        value=0.75  # 25% reduction
    )
    return tech


def _create_green_transition(isic_map):
    """Create multiple simultaneous green technology changes."""
    tech = TechnologicalChange(
        name="Green Technology Transition",
        description="Economy-wide efficiency improvements"
    )
    
    # 1. Energy efficiency across all sectors (assume energy is sector 1)
    energy_sector = _find_sector_by_keyword(isic_map, "energy", fallback=1)
    tech.add_input_change(
        input_sector_idx=energy_sector,
        change_type="multiply",
        value=0.75  # 25% reduction
    )
    
    # 2. Material efficiency in manufacturing (sectors 2 and 3)
    for sector in [2, 3]:
        tech.add_sector_change(
            sector_idx=sector,
            change_type="multiply",
            value=0.85,  # 15% reduction in all inputs
            exclude_inputs=[]
        )
    
    # 3. Services productivity (sector 4)
    tech.add_sector_change(
        sector_idx=4,
        change_type="multiply",
        value=0.90  # 10% reduction
    )
    
    return tech


def _create_custom_changes():
    """Create custom fine-grained changes."""
    tech = TechnologicalChange(
        name="Custom Technical Coefficient Changes",
        description="Specific targeted improvements"
    )
    
    # Multiple specific changes
    tech.add_coefficient_change(sector_idx=2, input_sector_idx=0, change_type="multiply", value=0.70)
    tech.add_coefficient_change(sector_idx=2, input_sector_idx=1, change_type="multiply", value=0.80)
    tech.add_coefficient_change(sector_idx=3, input_sector_idx=2, change_type="add", value=0.05)
    
    return tech


def _create_multi_level_production_changes():
    """
    Create a multi-level technological change demonstrating production-level modifications.
    
    This changes underlying production records rather than just the A matrix coefficients.
    The matrix will be rebuilt from the modified productions.
    """
    tech = TechnologicalChange(
        name="Production-Level Modernization",
        description="Firm-level technology adoption affecting underlying production methods"
    )
    
    # Level 2: Production-level changes
    # These require the matrix to be rebuilt from modified production records
    # Note: production_id refers to database IDs, which may vary by setup
    # For a demo, we use IDs 1-5 assuming a basic setup
    
    # Production 1: Reduce all input costs by 10% (process improvement)
    tech.add_production_all_inputs_change(
        production_id=1,
        change_type="multiply",
        value=0.90  # 10% cost reduction
    )
    
    # Production 2: Improve material efficiency
    tech.add_production_efficiency_change(
        production_id=2,
        efficiency_type="material",
        change_type="add",
        value=15  # +15 efficiency points
    )
    
    # Also add a matrix-level change to show mixed levels
    # Level 1: Additional productivity improvement in sector 3
    tech.add_sector_change(
        sector_idx=3,
        change_type="multiply",
        value=0.95  # 5% reduction in all inputs
    )
    
    return tech


def _create_capacity_expansion():
    """
    Create a curve-level technological change demonstrating supply capacity modifications.
    
    This changes the supply curve tiers (capacity and prices) for a sector.
    """
    tech = TechnologicalChange(
        name="Capacity Expansion with Cost Reduction",
        description="New production capacity comes online at lower cost"
    )
    
    # Level 3: Curve-level changes
    # Add new low-cost capacity tier to sector A01
    tech.add_curve_new_tier(
        isic="A01",
        cap=100,       # 100 units of new capacity
        price=35.0,    # At lower price than existing
        position=0     # Insert at beginning (cheapest tier)
    )
    
    # Expand existing capacity in sector A02
    tech.add_curve_tier_change(
        isic="A02",
        tier_index=0,
        field="cap",
        change_type="multiply",
        value=1.5  # 50% capacity increase
    )
    
    # Reduce prices across all tiers in sector A03 (technology improvement)
    tech.add_curve_scale_all_tiers(
        isic="A03",
        field="price",
        change_type="multiply",
        value=0.85  # 15% price reduction
    )
    
    return tech


def _create_green_tech_with_carbon_tax():
    """
    Create a green technology transition combined with carbon tax policy.
    
    This demonstrates how to combine:
    - Technological change (energy efficiency improvements)
    - Tax policy (increased corporate tax to represent carbon pricing)
    """
    tech = TechnologicalChange(
        name="Green Transition + Carbon Tax",
        description="Energy efficiency improvements with carbon tax funding transition"
    )
    
    # Level 1: Energy efficiency improvements
    # Reduce energy inputs across all sectors (assume energy is sector 1)
    tech.add_input_change(
        input_sector_idx=1,
        change_type="multiply",
        value=0.70  # 30% reduction through efficiency
    )
    
    # Level 1: Material efficiency in manufacturing (sector 2)
    tech.add_sector_change(
        sector_idx=2,
        change_type="multiply",
        value=0.85,  # 15% reduction
        exclude_inputs=[]
    )
    
    return tech


# Technological Change Examples Configuration
TECH_CHANGE_EXAMPLES = {
    1: {
        "title": "Energy Efficiency Improvement (30% reduction)",
        "description": [
            "Simulate a technological breakthrough that reduces energy consumption",
            "by 30% across ALL sectors. This could represent:",
            "  - More efficient equipment and machinery",
            "  - Better insulation and process optimization",
            "  - LED lighting, smart controls, etc.",
            "",
            "We compare the SAME final demand before and after the tech change."
        ],
        "tech_change_builder": lambda isic_map: create_energy_efficiency_change(
            energy_sector_idx=_find_sector_by_keyword(isic_map, "energy", fallback=1),
            efficiency_gain=0.30
        ),
        "final_demand": [250.0, 200.0, 300.0, 150.0, 100.0, 0.0]
    },
    
    2: {
        "title": "Manufacturing Productivity Improvement (20% all inputs)",
        "description": [
            "Sector [2] improves its production process, reducing ALL inputs by 20%.",
            "This represents general productivity gains from:",
            "  - Better management practices",
            "  - Process optimization",
            "  - Worker training and skill development",
            "  - Lean manufacturing techniques"
        ],
        "tech_change_builder": lambda isic_map: create_productivity_improvement(
            sector_idx=2,
            productivity_gain=0.20
        ),
        "final_demand": [200.0, 200.0, 200.0, 200.0, 200.0, 0.0]
    },
    
    3: {
        "title": "Substitution: Reduce Labor, Increase Capital (Automation)",
        "description": [
            "Sector [3] automates part of its production:",
            "  - Reduces input from sector [0] (e.g., labor services) by 40%",
            "  - Increases input from sector [2] (e.g., machinery) by 15%",
            "",
            "This represents automation/robotics replacing human workers."
        ],
        "tech_change_builder": lambda isic_map: _create_automation_substitution(),
        "final_demand": [180.0, 200.0, 250.0, 170.0, 200.0, 0.0]
    },
    
    4: {
        "title": "Material Efficiency: Reduce Waste in Production",
        "description": [
            "Sector [4] implements waste reduction techniques, decreasing",
            "material inputs from sector [1] by 25% through:",
            "  - Better quality control (less scrap)",
            "  - Precision manufacturing (less waste)",
            "  - Recycling and material recovery"
        ],
        "tech_change_builder": lambda isic_map: _create_material_efficiency(),
        "final_demand": [200.0, 180.0, 220.0, 200.0, 200.0, 0.0]
    },
    
    5: {
        "title": "Economy-Wide Green Technology Transition",
        "description": [
            "Multiple technological changes applied simultaneously:",
            "  - 25% energy efficiency improvement (all sectors)",
            "  - 15% material efficiency in manufacturing sectors [2,3]",
            "  - 10% productivity gain in services sector [4]",
            "",
            "Models a comprehensive green technology transition."
        ],
        "tech_change_builder": lambda isic_map: _create_green_transition(isic_map),
        "final_demand": [250.0, 220.0, 280.0, 180.0, 170.0, 0.0]
    },
    
    6: {
        "title": "Custom Scenario: Specific Coefficient Changes",
        "description": [
            "Demonstrates fine-grained control over technical coefficients.",
            "Example: Modify specific input-output relationships:",
            "  - Sector [2] input from [0]: multiply by 0.70 (30% reduction)",
            "  - Sector [2] input from [1]: multiply by 0.80 (20% reduction)",
            "  - Sector [3] input from [2]: add 0.05 (increase by fixed amount)"
        ],
        "tech_change_builder": lambda isic_map: _create_custom_changes(),
        "final_demand": [190.0, 210.0, 240.0, 160.0, 200.0, 0.0]
    },
    
    # === Multi-Level Examples (Production + Curve Level Changes) ===
    
    7: {
        "title": "Multi-Level: Production-Level Modernization",
        "description": [
            "LEVEL 2 (Production) + LEVEL 1 (Matrix) changes combined:",
            "",
            "This example modifies underlying PRODUCTION RECORDS, requiring",
            "the A matrix to be rebuilt from the modified data. Changes include:",
            "  - Production #1: 10% input cost reduction (process improvement)",
            "  - Production #2: +15 material efficiency points",
            "  - Sector [3]: 5% productivity gain (matrix level)",
            "",
            "Note: Production IDs depend on your database setup."
        ],
        "tech_change_builder": lambda isic_map: _create_multi_level_production_changes(),
        "final_demand": [200.0, 200.0, 200.0, 200.0, 200.0, 0.0],
        "use_multi_level": True  # Flag for demo.py to use tech_change parameter
    },
    
    8: {
        "title": "Multi-Level: Capacity Expansion with Cost Reduction",
        "description": [
            "LEVEL 3 (Curve) changes - Modifies supply curve tiers:",
            "",
            "This example changes the SUPPLY CURVES, affecting the",
            "price/capacity structure used by the equilibrium solver:",
            "  - A01: Add new 100-unit tier at $35 (cheapest)",
            "  - A02: 50% capacity increase in tier 0",
            "  - A03: 15% price reduction across all tiers",
            "",
            "Best used with solver_type='supply_curves'."
        ],
        "tech_change_builder": lambda isic_map: _create_capacity_expansion(),
        "final_demand": [200.0, 200.0, 200.0, 200.0, 200.0, 0.0],
        "use_multi_level": True,
        "solver_type": "supply_curves"  # Curve changes need supply_curves solver
    },
    
    # === Combined Technology + Tax Policy Example ===
    
    9: {
        "title": "COMBINED: Green Transition + Carbon Tax Policy",
        "description": [
            "Demonstrates SIMULTANEOUS technological change AND tax policy:",
            "",
            "TECHNOLOGICAL CHANGE (Energy & Material Efficiency):",
            "  - 30% energy efficiency improvement (all sectors)",
            "  - 15% material efficiency in manufacturing (sector 2)",
            "",
            "TAX POLICY CHANGE (Carbon Tax Implementation):",
            "  - Corporate tax: 25% → 35% (carbon pricing on profits)",
            "  - Income tax: 15% → 20% (to fund green transition)",
            "",
            "This models a comprehensive climate policy combining:",
            "  1. Technology incentives/adoption (efficiency gains)",
            "  2. Carbon pricing (via tax increases)",
            "  3. Revenue recycling (government spending on green infrastructure)",
            "",
            "The simulation shows how tech improvements can offset",
            "economic impacts of carbon taxation."
        ],
        "tech_change_builder": lambda isic_map: _create_green_tech_with_carbon_tax(),
        "final_demand": [250.0, 200.0, 280.0, 180.0, 170.0, 0.0],
        "use_multi_level": True,
        # Tax policy parameters (the key addition!)
        "income_tax_rate_before": 0.15,
        "income_tax_rate_after": 0.20,  # 5% increase
        "corporate_tax_rate_before": 0.25,
        "corporate_tax_rate_after": 0.35,  # 10% increase (carbon tax)
        "income_tax_applies_to": "bonusWages",
        "iterations": 3,  # Show circular flow effects
        "consumption_proportions": [0.30, 0.25, 0.20, 0.15, 0.10, 0.0],
        "investment_proportions": [0.10, 0.15, 0.40, 0.25, 0.10, 0.0],  # Green investment
        "government_proportions": [0.20, 0.25, 0.25, 0.20, 0.10, 0.0]   # Green infrastructure
    }
}
