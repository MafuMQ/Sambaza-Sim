"""
Technological Change Module for Input-Output Analysis
======================================================

This module provides the TechnologicalChange class and predefined templates for modeling
technological change in an Input-Output framework. Technological change affects the
technical coefficients (production recipes) in the input-output table.

Key Concepts:
-------------
1. Technical Coefficients (A matrix): Amount of input i needed per unit of output j
2. Technological change modifies these coefficients (e.g., less labor, more capital)
3. Comparisons are made with the SAME final demand (FD) to isolate the pure effect

Example Use Cases:
------------------
- Automation: Reduce labor inputs, increase machinery/capital inputs
- Energy efficiency: Reduce energy inputs per unit of output
- Process improvement: Reduce material waste (reduce material coefficients)
- Productivity gains: Reduce all inputs proportionally (scale efficiency)

Usage:
------
    from Input_Output_Model.demos.util.technological_change import TechnologicalChange
    from Input_Output_Model.demos.util.simulation import run_simulation

    # Create a tech change
    tech = TechnologicalChange(name="Energy Efficiency", description="...")
    tech.add_input_change(input_sector_idx=1, change_type="multiply", value=0.70)

    # Apply to IO matrix
    A_changed, VA_changed = tech.apply(A_baseline, VA_baseline)

    # Run unified comparison (supports tech change + tax policy simultaneously)
    run_simulation(
        final_demand=fd,
        A_before=A_baseline, A_after=A_changed,
        VA_before=VA_baseline, VA_after=VA_changed,
        isic_map=isic_map,
        before_name="Baseline", after_name="Energy Efficiency"
    )
"""

import numpy as np
import logging
from typing import Dict, List, Tuple, Optional
from copy import deepcopy

logger = logging.getLogger(__name__)


class TechnologicalChange:
    """
    Represents a technological change that can be applied to an Input-Output table.
    
    Technological changes modify the technical coefficient matrix (A) which represents
    the production recipes - how much of each input is needed per unit of output.
    """
    
    def __init__(self, name: str, description: str = ""):
        """
        Initialize a technological change.
        
        Parameters:
        -----------
        name : str
            Name of the technological change (e.g., "Automation", "Energy Efficiency")
        description : str
            Detailed description of what this change represents
        """
        self.name = name
        self.description = description
        self.changes = []  # List of (sector, input_sector, change_type, parameters)
        
    def add_coefficient_change(self, 
                               sector_idx: int, 
                               input_sector_idx: int, 
                               change_type: str,
                               value: float):
        """
        Add a change to a specific technical coefficient.
        
        Parameters:
        -----------
        sector_idx : int
            Index of the producing sector (column in A matrix)
        input_sector_idx : int
            Index of the input sector (row in A matrix)
        change_type : str
            Type of change: 
            - "multiply": Multiply coefficient by value (e.g., 0.8 = 20% reduction)
            - "add": Add value to coefficient (e.g., -0.05 = reduce by 0.05 units)
            - "set": Set coefficient to specific value
        value : float
            The value to apply based on change_type
        """
        valid_types = ["multiply", "add", "set"]
        if change_type not in valid_types:
            raise ValueError(f"change_type must be one of {valid_types}")
        
        self.changes.append({
            'sector_idx': sector_idx,
            'input_sector_idx': input_sector_idx,
            'change_type': change_type,
            'value': value
        })
        
    def add_sector_change(self,
                         sector_idx: int,
                         change_type: str,
                         value: float,
                         exclude_inputs: List[int] = None):
        """
        Apply a change to ALL inputs for a given sector.
        
        Parameters:
        -----------
        sector_idx : int
            Index of the producing sector (column in A matrix)
        change_type : str
            Type of change: "multiply", "add", or "set"
        value : float
            The value to apply
        exclude_inputs : List[int], optional
            List of input sector indices to exclude from the change
        """
        self.changes.append({
            'sector_idx': sector_idx,
            'input_sector_idx': 'all',
            'change_type': change_type,
            'value': value,
            'exclude_inputs': exclude_inputs or []
        })
        
    def add_input_change(self,
                        input_sector_idx: int,
                        change_type: str,
                        value: float,
                        exclude_sectors: List[int] = None):
        """
        Apply a change to a specific input across ALL sectors that use it.
        
        Parameters:
        -----------
        input_sector_idx : int
            Index of the input sector (row in A matrix)
        change_type : str
            Type of change: "multiply", "add", or "set"
        value : float
            The value to apply
        exclude_sectors : List[int], optional
            List of producing sector indices to exclude from the change
        """
        self.changes.append({
            'sector_idx': 'all',
            'input_sector_idx': input_sector_idx,
            'change_type': change_type,
            'value': value,
            'exclude_sectors': exclude_sectors or []
        })
    
    def apply(self, A_matrix: np.ndarray, VA_vector: np.ndarray = None) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        """
        Apply the technological change to an A matrix (and optionally VA vector).
        
        Parameters:
        -----------
        A_matrix : np.ndarray
            The technical coefficient matrix (n x n)
        VA_vector : np.ndarray, optional
            The value added coefficient vector (n,)
            If provided, will be adjusted to maintain A + VA = 1
        
        Returns:
        --------
        A_new : np.ndarray
            Modified coefficient matrix
        VA_new : np.ndarray or None
            Modified VA vector (if VA_vector was provided)
        """
        A_new = A_matrix.copy()
        n = A_new.shape[0]
        
        logger.info(f"Applying technological change: {self.name}")
        
        for change in self.changes:
            sector_idx = change['sector_idx']
            input_sector_idx = change['input_sector_idx']
            change_type = change['change_type']
            value = change['value']
            
            # Determine which cells to modify
            if sector_idx == 'all' and input_sector_idx != 'all':
                # Apply to entire row (one input across all sectors)
                exclude = change.get('exclude_sectors', [])
                for j in range(n):
                    if j not in exclude:
                        A_new[input_sector_idx, j] = self._apply_change(
                            A_new[input_sector_idx, j], change_type, value
                        )
                        
            elif sector_idx != 'all' and input_sector_idx == 'all':
                # Apply to entire column (all inputs for one sector)
                exclude = change.get('exclude_inputs', [])
                for i in range(n):
                    if i not in exclude:
                        A_new[i, sector_idx] = self._apply_change(
                            A_new[i, sector_idx], change_type, value
                        )
                        
            elif sector_idx != 'all' and input_sector_idx != 'all':
                # Apply to specific cell
                A_new[input_sector_idx, sector_idx] = self._apply_change(
                    A_new[input_sector_idx, sector_idx], change_type, value
                )
            else:
                raise ValueError("Cannot apply change to all sectors and all inputs simultaneously")
        
        # Adjust VA to maintain A + VA = 1 (if VA provided)
        VA_new = None
        if VA_vector is not None:
            VA_new = VA_vector.copy()
            # For each sector (column), calculate new VA = 1 - sum(inputs)
            for j in range(n):
                total_inputs = np.sum(A_new[:, j])
                if total_inputs <= 1.0:
                    VA_new[j] = 1.0 - total_inputs
                else:
                    logger.warning(f"Sector {j}: Total inputs ({total_inputs:.4f}) exceed 1.0 after technological change!")
                    VA_new[j] = 0.0
        
        logger.info(f"Technological change '{self.name}' applied successfully")
        return A_new, VA_new
    
    def _apply_change(self, old_value: float, change_type: str, value: float) -> float:
        """Apply a single change to a coefficient value."""
        if change_type == "multiply":
            return old_value * value
        elif change_type == "add":
            return max(0.0, old_value + value)  # Ensure non-negative
        elif change_type == "set":
            return value
        else:
            raise ValueError(f"Unknown change_type: {change_type}")
    
    def get_summary(self) -> str:
        """Return a human-readable summary of the changes."""
        summary = [f"\nTechnological Change: {self.name}"]
        if self.description:
            summary.append(f"Description: {self.description}")
        summary.append(f"\nNumber of changes: {len(self.changes)}")
        summary.append("\nDetails:")
        
        for i, change in enumerate(self.changes, 1):
            sector = change['sector_idx']
            input_s = change['input_sector_idx']
            ctype = change['change_type']
            val = change['value']
            
            if sector == 'all':
                summary.append(f"  {i}. Input [{input_s}] across all sectors: {ctype} by {val}")
            elif input_s == 'all':
                summary.append(f"  {i}. Sector [{sector}] all inputs: {ctype} by {val}")
            else:
                summary.append(f"  {i}. Sector [{sector}], Input [{input_s}]: {ctype} by {val}")
        
        return "\n".join(summary)


# ==============================================================================
# Predefined Technological Change Templates
# ==============================================================================

def create_automation_change(sector_idx: int, labor_reduction: float = 0.20) -> TechnologicalChange:
    """
    Create a technological change representing automation.
    
    Reduces labor inputs (typically in VA) and may increase capital/machinery inputs.
    
    Parameters:
    -----------
    sector_idx : int
        Which sector to automate
    labor_reduction : float
        Fraction to reduce labor by (e.g., 0.20 = 20% reduction)
    """
    tech_change = TechnologicalChange(
        name=f"Automation (Sector {sector_idx})",
        description=f"Reduce labor by {labor_reduction*100:.0f}% through automation"
    )
    
    # Note: This affects the production recipe. In practice, you would modify
    # the value-added components or specific labor-intensive inputs.
    # This is a template - customize based on your sector structure.
    
    return tech_change


def create_energy_efficiency_change(energy_sector_idx: int, efficiency_gain: float = 0.30) -> TechnologicalChange:
    """
    Create a technological change representing energy efficiency improvements.
    
    Reduces energy inputs across all sectors that use energy.
    
    Parameters:
    -----------
    energy_sector_idx : int
        Index of the energy/electricity sector
    efficiency_gain : float
        Fraction to reduce energy use by (e.g., 0.30 = 30% reduction)
    """
    tech_change = TechnologicalChange(
        name="Energy Efficiency Improvement",
        description=f"Reduce energy use by {efficiency_gain*100:.0f}% across all sectors"
    )
    
    # Reduce energy input across all sectors
    tech_change.add_input_change(
        input_sector_idx=energy_sector_idx,
        change_type="multiply",
        value=(1.0 - efficiency_gain)
    )
    
    return tech_change


def create_productivity_improvement(sector_idx: int, productivity_gain: float = 0.15) -> TechnologicalChange:
    """
    Create a technological change representing general productivity improvement.
    
    Reduces ALL inputs proportionally (more efficient use of everything).
    
    Parameters:
    -----------
    sector_idx : int
        Which sector improves productivity
    productivity_gain : float
        Fraction to reduce all inputs by (e.g., 0.15 = 15% reduction)
    """
    tech_change = TechnologicalChange(
        name=f"Productivity Improvement (Sector {sector_idx})",
        description=f"Increase productivity by {productivity_gain*100:.0f}% (reduce all inputs)"
    )
    
    # Reduce all inputs for this sector
    tech_change.add_sector_change(
        sector_idx=sector_idx,
        change_type="multiply",
        value=(1.0 - productivity_gain)
    )
    
    return tech_change
