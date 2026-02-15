"""
Technological Change Example Configurations
============================================

Configuration dictionaries for technological change simulation examples.
Each example demonstrates different types of technological innovation and their economic impacts.

Key Principle: Hold final demand CONSTANT to isolate the pure effect of technological change.
"""

import logging
from Input_Output_Model.demos.util.technological_change import (
    TechnologicalChange,
    create_energy_efficiency_change,
    create_productivity_improvement
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
    }
}
