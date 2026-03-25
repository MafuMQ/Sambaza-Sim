"""
Technological Change Module for Input-Output Analysis
======================================================

This module provides multi-level technological change modeling for Input-Output frameworks.
Changes can be applied at three levels, ordered by depth:

**Level 1 - Matrix Level (High Level)**:
    Direct changes to the A matrix coefficients. Fastest, no rebuilding required.
    Use for: Quick what-if analysis, sensitivity testing.
    Rebuild chain: None

**Level 2 - Curves Level (Market Level)**:
    Changes to supply curve tiers (capacities, prices, value added by tier).
    Requires rebuilding the coefficient matrix from the modified supply/flow data.
    Use for: Capacity expansion, cost structure changes, market entry/exit.
    Rebuild chain: Curves → Coefficient Matrix

**Level 3 - Productions Level (Micro Level)**:
    Changes to underlying production records (inputs, efficiencies, value added).
    Requires rebuilding supply curves from modified productions, then rebuilding
    the coefficient matrix from those curves. Full cascade rebuild.
    Use for: Firm-level technology adoption, process improvements.
    Rebuild chain: Productions → Supply Curves → Coefficient Matrix

Data Flow (deepest to shallowest):
    Productions (firms) → Supply Curves (flow matrix) → Coefficient Matrix (A matrix)

Key Concepts:
-------------
1. Technical Coefficients (A matrix): Amount of input i needed per unit of output j
2. Supply Curves: Tiered supply with price/capacity steps per sector (the flow matrix)
3. Productions: Individual production methods with input requirements and VA components

Example Use Cases:
------------------
- Matrix Level: Reduce energy coefficients by 30% economy-wide
- Curves Level: New capacity comes online at lower price tier
- Productions Level: A specific firm adopts new machinery (better efficiency)

Usage:
------
    from core.tech_change import TechnologicalChange

    # === Level 1: Matrix-level change (direct, no rebuild) ===
    tech = TechnologicalChange(name="Energy Efficiency")
    tech.add_input_change(input_sector_idx=1, change_type="multiply", value=0.70)
    A_changed, VA_changed = tech.apply(A_baseline, VA_baseline)

    # === Level 2: Curve-level change (rebuilds coefficient matrix) ===
    tech = TechnologicalChange(name="Capacity Expansion")
    tech.add_curve_tier_change(
        isic="A01",
        tier_index=0,  # First tier
        field="cap",
        change_type="add",
        value=50  # Add 50 units of capacity
    )
    tech.add_curve_new_tier(
        isic="A01",
        cap=100,
        price=45.0,  # New tier at lower price
        position=0   # Insert at beginning (cheapest)
    )
    # Modify curves and rebuild coefficient matrix
    result = tech.apply_to_curves(scdb)
    A_new, VA_new = result['A_matrix'], result['VA_vector']

    # === Level 3: Production-level change (full cascade rebuild) ===
    tech = TechnologicalChange(name="Factory Modernization")
    tech.add_production_input_change(
        production_id=5,  # Specific production method
        input_isic="A01",
        change_type="multiply",
        value=0.85  # 15% reduction in this input
    )
    tech.add_production_efficiency_change(
        production_id=5,
        efficiency_type="material",
        change_type="add",
        value=10  # +10 efficiency points
    )
    # Full cascade: productions → supply curves → coefficient matrix
    result = tech.apply_to_productions(ptdb, scdb)
    A_new, VA_new = result['A_matrix'], result['VA_vector']
"""

import numpy as np
import logging
from typing import Dict, List, Tuple, Optional, Any, Union
from copy import deepcopy

logger = logging.getLogger(__name__)


# ==============================================================================
# Change Level Constants
# ==============================================================================

LEVEL_MATRIX = "matrix"        # Direct A matrix coefficient changes
LEVEL_PRODUCTION = "production"  # Changes to production records
LEVEL_CURVE = "curve"          # Changes to supply curve tiers


class TechnologicalChange:
    """
    Represents a technological change that can be applied at multiple levels:
    
    1. Matrix Level: Direct changes to A matrix coefficients (fastest, no rebuild)
    2. Curves Level: Changes to supply curve tiers (rebuilds coefficient matrix)
    3. Productions Level: Changes to production records (rebuilds curves → coefficient matrix)
    
    You can mix changes from different levels in the same TechnologicalChange object;
    they will be applied in cascade order: productions → curves → matrix.
    
    Data flow: Productions → Supply Curves → Coefficient Matrix
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
        
        # Level 1: Matrix-level changes (direct A matrix modifications)
        self.matrix_changes = []
        
        # Level 2: Curve-level changes (require coefficient matrix rebuild)
        self.curve_changes = []
        
        # Level 3: Production-level changes (require supply curve + coefficient matrix rebuild)
        self.production_changes = []
        
        # ISIC mapping (set when using ISIC codes instead of indices)
        self._isic_map = None
        
        # Capital investment requirements for this technological change
        # Maps sector (ISIC code or index) -> dollar amount of capital goods needed
        self.capital_requirements = {}
        self.investment_duration = 1  # number of iterations for investment phase
        
        # Backward compatibility alias
        self.changes = self.matrix_changes
    
    # ==========================================================================
    # Level 1: Matrix-Level Changes (High Level)
    # ==========================================================================
        
    def add_coefficient_change(self, 
                               sector_idx: Union[int, str], 
                               input_sector_idx: Union[int, str], 
                               change_type: str,
                               value: float):
        """
        Add a change to a specific technical coefficient.
        
        Parameters:
        -----------
        sector_idx : int or str
            Index of the producing sector (column in A matrix) OR ISIC code
        input_sector_idx : int or str
            Index of the input sector (row in A matrix) OR ISIC code
        change_type : str
            Type of change: 
            - "multiply": Multiply coefficient by value (e.g., 0.8 = 20% reduction)
            - "add": Add value to coefficient (e.g., -0.05 = reduce by 0.05 units)
            - "set": Set coefficient to specific value
        value : float
            The value to apply based on change_type
            
        Note:
        -----
        You can use ISIC codes (e.g., "A01") instead of numeric indices.
        If using ISIC codes, you must provide isic_map when calling apply().
        """
        valid_types = ["multiply", "add", "set"]
        if change_type not in valid_types:
            raise ValueError(f"change_type must be one of {valid_types}")
        
        self.matrix_changes.append({
            'level': LEVEL_MATRIX,
            'sector_idx': sector_idx,
            'input_sector_idx': input_sector_idx,
            'change_type': change_type,
            'value': value
        })
        
    def add_sector_change(self,
                         sector_idx: Union[int, str],
                         change_type: str,
                         value: float,
                         exclude_inputs: List[Union[int, str]] = None):
        """
        Apply a change to ALL inputs for a given sector.
        
        Parameters:
        -----------
        sector_idx : int or str
            Index of the producing sector (column in A matrix) OR ISIC code
        change_type : str
            Type of change: "multiply", "add", or "set"
        value : float
            The value to apply
        exclude_inputs : List[int or str], optional
            List of input sector indices or ISIC codes to exclude from the change
            
        Note:
        -----
        You can use ISIC codes (e.g., "A01") instead of numeric indices.
        If using ISIC codes, you must provide isic_map when calling apply().
        """
        self.matrix_changes.append({
            'level': LEVEL_MATRIX,
            'sector_idx': sector_idx,
            'input_sector_idx': 'all',
            'change_type': change_type,
            'value': value,
            'exclude_inputs': exclude_inputs or []
        })
        
    def add_input_change(self,
                        input_sector_idx: Union[int, str],
                        change_type: str,
                        value: float,
                        exclude_sectors: List[Union[int, str]] = None):
        """
        Apply a change to a specific input across ALL sectors that use it.
        
        Parameters:
        -----------
        input_sector_idx : int or str
            Index of the input sector (row in A matrix) OR ISIC code
        change_type : str
            Type of change: "multiply", "add", or "set"
        value : float
            The value to apply
        exclude_sectors : List[int or str], optional
            List of producing sector indices or ISIC codes to exclude from the change
            
        Note:
        -----
        You can use ISIC codes (e.g., "A01") instead of numeric indices.
        If using ISIC codes, you must provide isic_map when calling apply().
        """
        self.matrix_changes.append({
            'level': LEVEL_MATRIX,
            'sector_idx': 'all',
            'input_sector_idx': input_sector_idx,
            'change_type': change_type,
            'value': value,
            'exclude_sectors': exclude_sectors or []
        })
    
    # ==========================================================================
    # Level 3: Production-Level Changes (Micro Level) — deepest
    # Rebuild chain: Productions → Supply Curves → Coefficient Matrix
    # ==========================================================================
    
    def add_production_input_change(self,
                                    production_id: int,
                                    input_isic: str,
                                    change_type: str,
                                    value: float):
        """
        Change a specific input cost for a production method.
        
        This modifies the production_inputs field of a Production record.
        After applying, the A matrix must be rebuilt from the modified productions.
        
        Parameters:
        -----------
        production_id : int
            Database ID of the production method to modify
        input_isic : str
            ISIC code of the input to change (e.g., "A01", "B05")
        change_type : str
            Type of change: "multiply", "add", or "set"
        value : float
            The value to apply based on change_type
        """
        self.production_changes.append({
            'level': LEVEL_PRODUCTION,
            'production_id': production_id,
            'field': 'production_inputs',
            'input_isic': input_isic,
            'change_type': change_type,
            'value': value
        })
    
    def add_production_all_inputs_change(self,
                                         production_id: int,
                                         change_type: str,
                                         value: float,
                                         exclude_isics: List[str] = None):
        """
        Change ALL input costs for a production method uniformly.
        
        Parameters:
        -----------
        production_id : int
            Database ID of the production method to modify
        change_type : str
            Type of change: "multiply", "add", or "set"
        value : float
            The value to apply based on change_type
        exclude_isics : List[str], optional
            List of ISIC codes to exclude from the change
        """
        self.production_changes.append({
            'level': LEVEL_PRODUCTION,
            'production_id': production_id,
            'field': 'production_inputs',
            'input_isic': 'all',
            'change_type': change_type,
            'value': value,
            'exclude_isics': exclude_isics or []
        })
    
    def add_production_va_change(self,
                                 production_id: int,
                                 va_component: str,
                                 change_type: str,
                                 value: float):
        """
        Change a value-added component for a production method.
        
        Parameters:
        -----------
        production_id : int
            Database ID of the production method to modify
        va_component : str
            VA component to change: "minWages", "bonusWages", "surplus", or "total"
        change_type : str
            Type of change: "multiply", "add", or "set"
        value : float
            The value to apply based on change_type
        """
        valid_components = ["minWages", "bonusWages", "surplus", "total"]
        if va_component not in valid_components:
            raise ValueError(f"va_component must be one of {valid_components}")
        
        self.production_changes.append({
            'level': LEVEL_PRODUCTION,
            'production_id': production_id,
            'field': 'production_added_values',
            'va_component': va_component,
            'change_type': change_type,
            'value': value
        })
    
    def add_production_efficiency_change(self,
                                         production_id: int,
                                         efficiency_type: str,
                                         change_type: str,
                                         value: float):
        """
        Change an efficiency metric for a production method.
        
        Parameters:
        -----------
        production_id : int
            Database ID of the production method to modify
        efficiency_type : str
            Efficiency to change: "material", "labour", or "energy"
        change_type : str
            Type of change: "multiply", "add", or "set"
        value : float
            The value to apply based on change_type
        """
        efficiency_fields = {
            "material": "production_material_efficiency",
            "labour": "production_labour_efficiency",
            "energy": "production_energy_efficiency"
        }
        if efficiency_type not in efficiency_fields:
            raise ValueError(f"efficiency_type must be one of {list(efficiency_fields.keys())}")
        
        self.production_changes.append({
            'level': LEVEL_PRODUCTION,
            'production_id': production_id,
            'field': efficiency_fields[efficiency_type],
            'change_type': change_type,
            'value': value
        })
    
    def add_production_price_change(self,
                                    production_id: int,
                                    change_type: str,
                                    value: float):
        """
        Change the output price for a production method.
        
        Parameters:
        -----------
        production_id : int
            Database ID of the production method to modify
        change_type : str
            Type of change: "multiply", "add", or "set"
        value : float
            The value to apply based on change_type
        """
        self.production_changes.append({
            'level': LEVEL_PRODUCTION,
            'production_id': production_id,
            'field': 'price',
            'change_type': change_type,
            'value': value
        })
    
    # ==========================================================================
    # Level 2: Curve-Level Changes (Market Level)
    # Rebuild chain: Supply Curves → Coefficient Matrix
    # ==========================================================================
    
    def add_curve_tier_change(self,
                              isic: str,
                              tier_index: int,
                              field: str,
                              change_type: str,
                              value: float):
        """
        Change a specific field of a supply curve tier.
        
        Parameters:
        -----------
        isic : str
            ISIC code of the good's supply curve
        tier_index : int
            Index of the tier to modify (0 = first/cheapest tier)
        field : str
            Field to change: "cap", "price", or a VA component
        change_type : str
            Type of change: "multiply", "add", or "set"
        value : float
            The value to apply based on change_type
        """
        valid_fields = ["cap", "price", "minWages", "bonusWages", "surplus"]
        if field not in valid_fields:
            raise ValueError(f"field must be one of {valid_fields}")
        
        self.curve_changes.append({
            'level': LEVEL_CURVE,
            'isic': isic,
            'action': 'modify_tier',
            'tier_index': tier_index,
            'field': field,
            'change_type': change_type,
            'value': value
        })
    
    def add_curve_new_tier(self,
                           isic: str,
                           cap: float,
                           price: float,
                           position: int = -1,
                           va_components: Dict[str, float] = None):
        """
        Add a new tier to a supply curve.
        
        Parameters:
        -----------
        isic : str
            ISIC code of the good's supply curve
        cap : float
            Capacity of the new tier (use -1 for infinite)
        price : float
            Price of the new tier
        position : int
            Where to insert the tier (-1 = append at end, 0 = insert at beginning)
        va_components : dict, optional
            Value added components {"minWages": x, "bonusWages": y, "surplus": z}
        """
        self.curve_changes.append({
            'level': LEVEL_CURVE,
            'isic': isic,
            'action': 'add_tier',
            'cap': cap,
            'price': price,
            'position': position,
            'va_components': va_components or {}
        })
    
    def add_curve_remove_tier(self, isic: str, tier_index: int):
        """
        Remove a tier from a supply curve.
        
        Parameters:
        -----------
        isic : str
            ISIC code of the good's supply curve
        tier_index : int
            Index of the tier to remove
        """
        self.curve_changes.append({
            'level': LEVEL_CURVE,
            'isic': isic,
            'action': 'remove_tier',
            'tier_index': tier_index
        })
    
    def add_curve_scale_all_tiers(self,
                                   isic: str,
                                   field: str,
                                   change_type: str,
                                   value: float):
        """
        Apply a change to all tiers of a supply curve.
        
        Parameters:
        -----------
        isic : str
            ISIC code of the good's supply curve
        field : str
            Field to change: "cap" or "price"
        change_type : str
            Type of change: "multiply", "add", or "set"
        value : float
            The value to apply based on change_type
        """
        self.curve_changes.append({
            'level': LEVEL_CURVE,
            'isic': isic,
            'action': 'scale_all_tiers',
            'field': field,
            'change_type': change_type,
            'value': value
        })
    
    # ==========================================================================
    # Capital Requirements (Investment Cost of Technology Change)
    # ==========================================================================
    
    def set_capital_requirements(self, requirements: Dict[Union[str, int], float],
                                  investment_duration: int = 1):
        """
        Set the capital goods required to implement this technological change.
        
        This defines what must be purchased (as investment demand) BEFORE the
        technology change takes effect. The simulation runs in two phases:
        
        Phase 1 (Investment): Capital demand is injected into the economy.
            The OLD technology is still active. Capital-producing sectors
            receive extra demand to produce the required equipment/infrastructure.
        
        Phase 2 (New Technology): Capital has been delivered. The technology
            change takes effect (A matrix switches to A_after).
        
        Parameters:
        -----------
        requirements : dict
            Mapping of sector ISIC code (or index) to dollar amount of capital
            goods needed from that sector.
            Example: {"C28_281_2821": 500.0, "F41_411_4110": 200.0}
            meaning: need $500 of machinery and $200 of construction
        investment_duration : int
            Number of iterations the investment phase lasts (default: 1).
            The technology switches after this many iterations.
        """
        self.capital_requirements = requirements
        self.investment_duration = max(1, investment_duration)
    
    def has_capital_requirements(self) -> bool:
        """Check if this technological change requires capital investment."""
        return len(self.capital_requirements) > 0 and sum(self.capital_requirements.values()) > 0
    
    def get_total_capital_cost(self) -> float:
        """Get total capital investment required."""
        return sum(self.capital_requirements.values())
    
    def get_capital_demand_vector(self, isic_map: Dict[str, int], n: int) -> 'np.ndarray':
        """
        Convert capital requirements into a demand vector for the I-O model.
        
        Each entry represents the dollar amount of capital goods that must be
        produced by that sector to enable this technological change.
        
        Parameters:
        -----------
        isic_map : dict
            Mapping of ISIC codes to matrix indices
        n : int
            Number of sectors
        
        Returns:
        --------
        np.ndarray : Capital demand vector (n,) with amounts per sector
        """
        capital_demand = np.zeros(n)
        for sector, amount in self.capital_requirements.items():
            if isinstance(sector, str):
                if sector in isic_map:
                    capital_demand[isic_map[sector]] = amount
                else:
                    logger.warning(f"Capital requirement sector '{sector}' not found in isic_map")
            elif isinstance(sector, int):
                if 0 <= sector < n:
                    capital_demand[sector] = amount
                else:
                    logger.warning(f"Capital requirement sector index {sector} out of range (n={n})")
            else:
                logger.warning(f"Invalid sector reference in capital_requirements: {sector}")
        return capital_demand
    
    # ==========================================================================
    # Apply Methods
    # ==========================================================================
    
    def set_isic_map(self, isic_map: Dict[str, int]):
        """
        Set the ISIC to index mapping for resolving ISIC codes.
        
        Parameters:
        -----------
        isic_map : dict
            Mapping of ISIC codes to sector indices {"A01": 0, "A02": 1, ...}
        """
        self._isic_map = isic_map
    
    def _resolve_index(self, value: Union[int, str]) -> int:
        """
        Resolve a sector reference to an integer index.
        
        Parameters:
        -----------
        value : int or str
            Either an integer index or an ISIC code
            
        Returns:
        --------
        int : The sector index
        """
        if isinstance(value, int):
            return value
        elif isinstance(value, str):
            if self._isic_map is None:
                raise ValueError(f"Cannot resolve ISIC code '{value}': isic_map not set. Call set_isic_map() or pass isic_map to apply().")
            if value not in self._isic_map:
                raise ValueError(f"ISIC code '{value}' not found in isic_map. Available: {list(self._isic_map.keys())}")
            return self._isic_map[value]
        else:
            raise TypeError(f"Sector reference must be int or str, got {type(value)}")
    
    def apply(self, A_matrix: np.ndarray, VA_vector: np.ndarray = None, isic_map: Dict[str, int] = None) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        """
        Apply matrix-level technological changes to an A matrix (and optionally VA vector).
        
        This method only applies LEVEL 1 (matrix-level) changes. For curve-level
        or production-level changes, use apply_to_curves() or apply_to_productions().
        
        Parameters:
        -----------
        A_matrix : np.ndarray
            The technical coefficient matrix (n x n)
        VA_vector : np.ndarray, optional
            The value added coefficient vector (n,)
            If provided, will be adjusted to maintain A + VA = 1
        isic_map : dict, optional
            Mapping of ISIC codes to sector indices {"A01": 0, "A02": 1, ...}
            Required if any changes use ISIC codes instead of numeric indices
        
        Returns:
        --------
        A_new : np.ndarray
            Modified coefficient matrix
        VA_new : np.ndarray or None
            Modified VA vector (if VA_vector was provided)
        """
        # Store isic_map if provided
        if isic_map is not None:
            self._isic_map = isic_map
        
        A_new = A_matrix.copy()
        n = A_new.shape[0]
        
        logger.info(f"Applying matrix-level technological change: {self.name}")
        
        for change in self.matrix_changes:
            sector_idx = change['sector_idx']
            input_sector_idx = change['input_sector_idx']
            change_type = change['change_type']
            value = change['value']
            
            # Resolve ISIC codes to indices if needed
            if sector_idx != 'all':
                sector_idx = self._resolve_index(sector_idx)
            if input_sector_idx != 'all':
                input_sector_idx = self._resolve_index(input_sector_idx)
            
            # Resolve exclude lists
            exclude_sectors = [self._resolve_index(x) for x in change.get('exclude_sectors', [])]
            exclude_inputs = [self._resolve_index(x) for x in change.get('exclude_inputs', [])]
            
            # Determine which cells to modify
            if sector_idx == 'all' and input_sector_idx != 'all':
                # Apply to entire row (one input across all sectors)
                for j in range(n):
                    if j not in exclude_sectors:
                        A_new[input_sector_idx, j] = self._apply_change(
                            A_new[input_sector_idx, j], change_type, value
                        )
                        
            elif sector_idx != 'all' and input_sector_idx == 'all':
                # Apply to entire column (all inputs for one sector)
                for i in range(n):
                    if i not in exclude_inputs:
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
    
    def apply_to_productions(self, ptdb, scdb=None, rebuild_curves: bool = True,
                              rebuild_matrix: bool = True,
                              loggingLevel=logging.WARNING) -> Dict[str, Any]:
        """
        Apply production-level changes to a ProductionsDatabase.
        
        Level 3 rebuild chain: Productions → Supply Curves → Coefficient Matrix
        
        This modifies production records and cascades the rebuild:
        1. Modified productions are used to rebuild supply curves
        2. Modified supply curves are used to rebuild the coefficient matrix
        
        Changes are applied in-memory; the database is NOT permanently modified.
        
        Parameters:
        -----------
        ptdb : ProductionsDatabase
            The productions database to read from
        scdb : SupplyCurveDatabase, optional
            Supply curve database (required for rebuild)
        rebuild_curves : bool
            Whether to rebuild supply curves from modified productions (default: True)
        rebuild_matrix : bool
            Whether to rebuild the coefficient matrix from rebuilt curves (default: True)
            Only applies if rebuild_curves is also True.
        loggingLevel : int
            Logging level for rebuild operations
        
        Returns:
        --------
        dict with:
            'productions': dict mapping production_id -> modified production data
            'supply_data': rebuilt supply curve data (if rebuild_curves=True)
            'A_matrix': rebuilt A matrix (if rebuild_matrix=True)
            'VA_vector': rebuilt VA vector (if rebuild_matrix=True)
            'isic_map': sector index mapping (if rebuild_matrix=True)
        """
        if not self.production_changes:
            logger.warning(f"No production-level changes to apply for '{self.name}'")
            return {'productions': {}, 'A_matrix': None, 'VA_vector': None, 'isic_map': None}
        
        logger.info(f"Applying {len(self.production_changes)} production-level changes: {self.name}")
        
        # Load all productions that will be modified
        modified_productions = {}
        
        for change in self.production_changes:
            prod_id = change['production_id']
            
            # Get production if not already loaded
            if prod_id not in modified_productions:
                prod = ptdb.get_production_by_id(prod_id)
                if prod is None:
                    logger.warning(f"Production ID {prod_id} not found, skipping")
                    continue
                # Create a mutable copy of the production data
                modified_productions[prod_id] = {
                    'id': prod.id,
                    'name': prod.name,
                    'id_number': prod.id_number,
                    'isic': prod.isic,
                    'production_inputs': deepcopy(prod.production_inputs) or {},
                    'production_added_values': deepcopy(prod.production_added_values) or {},
                    'price': float(prod.price) if prod.price else 0.0,
                    'total_value_added': float(prod.total_value_added) if prod.total_value_added else 0.0,
                    'total_inputs_cost': float(prod.total_inputs_cost) if prod.total_inputs_cost else 0.0,
                    'production_material_efficiency': prod.production_material_efficiency or 0,
                    'production_labour_efficiency': prod.production_labour_efficiency or 0,
                    'production_energy_efficiency': prod.production_energy_efficiency or 0,
                }
            
            prod_data = modified_productions[prod_id]
            field = change['field']
            change_type = change['change_type']
            value = change['value']
            
            # Apply the change based on field type
            if field == 'production_inputs':
                input_isic = change.get('input_isic')
                if input_isic == 'all':
                    # Apply to all inputs
                    exclude = change.get('exclude_isics', [])
                    for isic in list(prod_data['production_inputs'].keys()):
                        if isic not in exclude:
                            old_val = float(prod_data['production_inputs'].get(isic, 0))
                            prod_data['production_inputs'][isic] = self._apply_change(old_val, change_type, value)
                else:
                    # Apply to specific input
                    old_val = float(prod_data['production_inputs'].get(input_isic, 0))
                    prod_data['production_inputs'][input_isic] = self._apply_change(old_val, change_type, value)
                    
            elif field == 'production_added_values':
                va_component = change.get('va_component')
                if va_component == 'total':
                    # Scale all VA components proportionally
                    for comp in ['minWages', 'bonusWages', 'surplus']:
                        if comp in prod_data['production_added_values']:
                            old_val = float(prod_data['production_added_values'][comp])
                            prod_data['production_added_values'][comp] = self._apply_change(old_val, change_type, value)
                else:
                    old_val = float(prod_data['production_added_values'].get(va_component, 0))
                    prod_data['production_added_values'][va_component] = self._apply_change(old_val, change_type, value)
                    
            elif field == 'price':
                prod_data['price'] = self._apply_change(prod_data['price'], change_type, value)
                
            elif field in ['production_material_efficiency', 'production_labour_efficiency', 'production_energy_efficiency']:
                old_val = prod_data[field] or 0
                prod_data[field] = int(self._apply_change(old_val, change_type, value))
            
            # Recalculate derived fields
            prod_data['total_inputs_cost'] = sum(float(v) for v in prod_data['production_inputs'].values())
            prod_data['total_value_added'] = sum(float(v) for v in prod_data['production_added_values'].values())
        
        result = {
            'productions': modified_productions, 
            'supply_data': None,
            'A_matrix': None, 
            'VA_vector': None, 
            'isic_map': None
        }
        
        # Cascade rebuild: Productions → Supply Curves → Coefficient Matrix
        if rebuild_curves:
            curves_result = self._rebuild_curves_from_productions(ptdb, scdb, modified_productions, loggingLevel)
            result['supply_data'] = curves_result.get('supply_data')
            
            if rebuild_matrix:
                # Rebuild coefficient matrix from the rebuilt supply curves
                matrix_result = self._rebuild_matrix_from_curves(
                    scdb, ptdb, curves_result.get('supply_data', {}), loggingLevel
                )
                result.update(matrix_result)
                logger.info(f"Cascade rebuild complete: productions → curves → coefficient matrix")
        
        logger.info(f"Production-level changes applied: {len(modified_productions)} productions modified")
        return result
    
    def _rebuild_curves_from_productions(self, ptdb, scdb, modified_productions: Dict,
                                           loggingLevel=logging.WARNING) -> Dict[str, Any]:
        """
        Rebuild supply curve data from production records, incorporating modifications.
        
        This is the first step of the Level 3 cascade:
        Productions → **Supply Curves** → Coefficient Matrix
        
        Matches the existing project's curve builder structure:
        - Sorts productions by merit order (cheapest price first)
        - Creates three parallel tier lists: price, total_inputs_cost, total_value_added
        - Each tier: {'cap': x, 'price': y}
        
        Parameters:
        -----------
        ptdb : ProductionsDatabase
            The productions database
        scdb : SupplyCurveDatabase
            The supply curve database (for structure/ordering)
        modified_productions : dict
            Mapping of production_id -> modified production data
        loggingLevel : int
            Logging level
        
        Returns:
        --------
        dict with:
            'supply_data': dict mapping ISIC -> curve structure matching existing format
        """
        from Input_Output_Model.models.entities.SupplyCurve import SupplyCurveDatabase
        
        if scdb is None:
            scdb = SupplyCurveDatabase()
        
        # Get supply curves to determine structure
        supply_curves = scdb.get_all_supply_curves()
        sorted_curves = sorted(supply_curves, key=lambda x: x.isic)
        
        supply_data = {}
        
        for curve in sorted_curves:
            isic = curve.isic
            
            # Get all productions for this good
            prods = ptdb.get_all_productions_by_good(int(curve.id_number))
            if not prods:
                # No productions — use empty tiers
                supply_data[isic] = {
                    "price": {"tiers": []},
                    "total_inputs_cost": {"tiers": []},
                    "total_value_added": {"tiers": []}
                }
                continue
            
            # Build function dicts for each production (modified or original)
            function_dicts = []
            for p in prods:
                if p.id in modified_productions:
                    # Use modified production data
                    pd = modified_productions[p.id]
                    func_dict = {
                        'cap': float(p.production_quantity) if p.production_quantity else -1,
                        'price': float(pd['price']),
                        'total_inputs_cost': float(pd['total_inputs_cost']),
                        'total_value_added': float(pd['total_value_added']),
                    }
                else:
                    # Use original production data
                    func_dict = {
                        'cap': float(p.production_quantity) if p.production_quantity else -1,
                        'price': float(p.price) if p.price else 0.0,
                        'total_inputs_cost': float(p.total_inputs_cost) if p.total_inputs_cost else 0.0,
                        'total_value_added': float(p.total_value_added) if p.total_value_added else 0.0,
                    }
                function_dicts.append(func_dict)
            
            # Sort by price (merit order - cheapest first)
            sorted_prods = sorted(function_dicts, key=lambda x: x['price'])
            
            # Build the three parallel tier lists (matching existing format)
            price_tiers = []
            input_cost_tiers = []
            value_added_tiers = []
            
            for func_dict in sorted_prods:
                cap = func_dict['cap']
                
                price_tiers.append({
                    "cap": cap,
                    "price": func_dict['price']
                })
                
                input_cost_tiers.append({
                    "cap": cap,
                    "price": func_dict['total_inputs_cost']
                })
                
                value_added_tiers.append({
                    "cap": cap,
                    "price": func_dict['total_value_added']
                })
            
            # Store in the format matching existing curve structure
            supply_data[isic] = {
                "price": {"tiers": price_tiers},
                "total_inputs_cost": {"tiers": input_cost_tiers},
                "total_value_added": {"tiers": value_added_tiers}
            }
        
        logger.info(f"Supply curves rebuilt from productions: {len(supply_data)} goods")
        return {'supply_data': supply_data}
    
    def _rebuild_matrix_from_curves(self, scdb, ptdb, supply_data: Dict,
                                     loggingLevel=logging.WARNING) -> Dict[str, Any]:
        """
        Rebuild the A matrix and VA vector from supply curve / flow data.
        
        This is the final step of the rebuild cascade:
        Productions → Supply Curves → **Coefficient Matrix**
        
        Key insight: Supply curves store aggregate costs (price, total_inputs_cost, 
        total_value_added) but NOT the detailed input breakdown. The input breakdown 
        must be fetched from the productions database.
        
        Algorithm:
        1. For each good (output ISIC), find cheapest tier from curves
        2. Fetch that production's detailed input breakdown from productions database
        3. Calculate input coefficients: input_cost / output_price
        4. Build coefficient matrix A where A[input_row, output_col] = coefficient
        
        Parameters:
        -----------
        scdb : SupplyCurveDatabase
            The supply curve database (for structure)
        ptdb : ProductionsDatabase
            Productions database  (required for detailed input coefficients)
        supply_data : dict
            Mapping of ISIC -> curve structure from supply curves or rebuild
            Format: {"price": {"tiers": [...]}, "total_inputs_cost": {"tiers": [...]}, ...}
        loggingLevel : int
            Logging level
        
        Returns:
        --------
        dict with:
            'A_matrix': rebuilt coefficient matrix (n x n)
            'VA_vector': rebuilt VA vector (n,)
            'isic_map': ISIC -> index mapping
        """
        from Input_Output_Model.models.entities.SupplyCurve import SupplyCurveDatabase
        
        if scdb is None:
            scdb = SupplyCurveDatabase()
        
        # Get supply curves for structure ordering
        all_curves = scdb.get_all_supply_curves()
        sorted_curves = sorted(all_curves, key=lambda x: x.isic)
        
        n = len(sorted_curves)
        isic_map = {curve.isic: i for i, curve in enumerate(sorted_curves)}
        
        # Initialize matrices
        A_mon = np.zeros((n, n))
        VA_mon = np.zeros(n)
        
        for curve in sorted_curves:
            isic = curve.isic
            col_idx = isic_map[isic]
            
            # Get curve data for this good from supply_data
            curve_data = supply_data.get(isic)
            if not curve_data:
                continue
            
            # Extract price tiers (cheapest first due to merit order sorting)
            if isinstance(curve_data, dict) and 'price' in curve_data:
                price_tiers = curve_data['price'].get('tiers', [])
                va_tiers = curve_data.get('total_value_added', {}).get('tiers', [])
            else:
                # Fallback for other formats
                price_tiers = curve_data if isinstance(curve_data, list) else []
                va_tiers = []
            
            if not price_tiers:
                continue
            
            # Use the cheapest tier (first tier)
            cheapest_tier = price_tiers[0]
            output_price = cheapest_tier.get('price', 0.0)
            
            if output_price <= 0:
                output_price = 1.0
            
            # Get VA from tiers
            if va_tiers:
                va_value = va_tiers[0].get('price', 0.0)  # 'price' field stores the VA value
                VA_mon[col_idx] = float(va_value) / output_price
            
            # Get detailed input breakdown from productions database
            # (Supply curves don't store this — only aggregate costs)
            if ptdb is None:
                continue
            
            prods = ptdb.get_all_productions_by_good(int(curve.id_number))
            if not prods:
                continue
            
            # Find the production matching the cheapest price
            matching_prod = None
            min_price_diff = float('inf')
            for p in prods:
                p_price = float(p.price) if p.price else 0.0
                price_diff = abs(p_price - output_price)
                if price_diff < min_price_diff:
                    min_price_diff = price_diff
                    matching_prod = p
            
            if not matching_prod or not matching_prod.production_inputs:
                continue
            
            # Build the coefficient column from this production's inputs
            for input_isic, input_cost in matching_prod.production_inputs.items():
                if input_isic in isic_map:
                    row_idx = isic_map[input_isic]
                    A_mon[row_idx, col_idx] = float(input_cost) / output_price
        
        logger.info(f"Coefficient matrix rebuilt from supply curves: {n}x{n}")
        return {'A_matrix': A_mon, 'VA_vector': VA_mon, 'isic_map': isic_map}
    
    def apply_to_curves(self, scdb, ptdb=None, isic_map: Dict[str, int] = None,
                         rebuild_matrix: bool = True,
                         loggingLevel=logging.WARNING) -> Dict[str, Any]:
        """
        Apply curve-level changes to supply curve data and rebuild the coefficient matrix.
        
        Level 2 rebuild chain: Curves → Coefficient Matrix
        
        This modifies supply curve tiers and then rebuilds the A matrix and VA vector
        from the modified flow data.
        
        Parameters:
        -----------
        scdb : SupplyCurveDatabase
            The supply curve database to read from
        ptdb : ProductionsDatabase, optional
            Productions database (required if rebuild_matrix=True)
        isic_map : dict, optional
            Mapping of ISIC codes to sector indices
        rebuild_matrix : bool
            Whether to rebuild A matrix from modified curves (default: True)
        loggingLevel : int
            Logging level for matrix build operations
        
        Returns:
        --------
        dict with:
            'supply_data': dict mapping ISIC -> modified tier list
            'modified_curves': dict of modified curve data
            'A_matrix': rebuilt coefficient matrix (if rebuild_matrix=True)
            'VA_vector': rebuilt VA vector (if rebuild_matrix=True)
            'isic_map': sector index mapping (if rebuild_matrix=True)
        """
        if not self.curve_changes:
            logger.warning(f"No curve-level changes to apply for '{self.name}'")
            return {'supply_data': {}, 'modified_curves': {}}
        
        logger.info(f"Applying {len(self.curve_changes)} curve-level changes: {self.name}")
        
        # Load all supply curves
        all_curves = scdb.get_all_supply_curves()
        
        # Build initial supply data structure
        supply_data = {}
        curve_by_isic = {}
        
        for curve in all_curves:
            curve_by_isic[curve.isic] = curve
            
            # Parse tiers from the supply curve data
            # Data format: curve.price is a list of {'cap': x, 'price': y} dicts
            # curve.total_inputs_cost is a parallel list of {'cap': x, 'price': y} (input costs)
            # curve.total_value_added is a parallel list of {'cap': x, 'price': y} (VA amounts)
            price_data = curve.price if isinstance(curve.price, list) else []
            ic_data = curve.total_inputs_cost if isinstance(curve.total_inputs_cost, list) else []
            va_data = curve.total_value_added if isinstance(curve.total_value_added, list) else []
            
            if price_data:
                tiers = []
                for i, entry in enumerate(price_data):
                    if isinstance(entry, dict):
                        # Standard format: {'cap': x, 'price': y}
                        tier = {
                            'price': float(entry.get('price', 0)),
                            'cap': float(entry.get('cap', -1)),
                        }
                    else:
                        # Fallback: plain number
                        tier = {
                            'price': float(entry),
                            'cap': -1,
                        }
                    
                    # Add input cost data if available
                    if i < len(ic_data):
                        ic_entry = ic_data[i]
                        if isinstance(ic_entry, dict):
                            tier['total_inputs_cost'] = float(ic_entry.get('price', 0))
                        else:
                            tier['total_inputs_cost'] = float(ic_entry)
                    
                    # Add VA data if available
                    if i < len(va_data):
                        va_entry = va_data[i]
                        if isinstance(va_entry, dict):
                            tier['total_va'] = float(va_entry.get('price', 0))
                        else:
                            tier['total_va'] = float(va_entry)
                    
                    # Add VA component breakdown if available
                    if curve.production_added_values:
                        pav = curve.production_added_values
                        if isinstance(pav, list) and i < len(pav):
                            tier.update(pav[i])
                        elif isinstance(pav, dict):
                            tier.update(pav)
                    
                    tiers.append(tier)
                
                supply_data[curve.isic] = tiers
        
        # Apply changes
        modified_curves = set()
        
        for change in self.curve_changes:
            isic = change['isic']
            action = change['action']
            
            if isic not in supply_data:
                logger.warning(f"ISIC {isic} not found in supply curves, skipping")
                continue
            
            modified_curves.add(isic)
            tiers = supply_data[isic]
            
            if action == 'modify_tier':
                tier_idx = change['tier_index']
                if tier_idx < 0 or tier_idx >= len(tiers):
                    logger.warning(f"Tier index {tier_idx} out of range for {isic}, skipping")
                    continue
                
                field = change['field']
                old_val = float(tiers[tier_idx].get(field, 0))
                tiers[tier_idx][field] = self._apply_change(old_val, change['change_type'], change['value'])
                
            elif action == 'add_tier':
                new_tier = {
                    'price': change['price'],
                    'cap': change['cap']
                }
                if change.get('va_components'):
                    new_tier.update(change['va_components'])
                
                position = change.get('position', -1)
                if position < 0 or position >= len(tiers):
                    tiers.append(new_tier)
                else:
                    tiers.insert(position, new_tier)
                    
            elif action == 'remove_tier':
                tier_idx = change['tier_index']
                if 0 <= tier_idx < len(tiers) and len(tiers) > 1:
                    tiers.pop(tier_idx)
                else:
                    logger.warning(f"Cannot remove tier {tier_idx} from {isic}")
                    
            elif action == 'scale_all_tiers':
                field = change['field']
                for tier in tiers:
                    if field in tier:
                        old_val = float(tier[field])
                        tier[field] = self._apply_change(old_val, change['change_type'], change['value'])
        
        logger.info(f"Curve-level changes applied: {len(modified_curves)} curves modified")
        
        result = {
            'supply_data': supply_data, 
            'modified_curves': {isic: supply_data[isic] for isic in modified_curves},
            'A_matrix': None,
            'VA_vector': None,
            'isic_map': None
        }
        
        # Rebuild coefficient matrix from modified supply curves
        if rebuild_matrix:
            matrix_result = self._rebuild_matrix_from_curves(scdb, ptdb, supply_data, loggingLevel)
            result.update(matrix_result)
            logger.info(f"Coefficient matrix rebuilt from modified supply curves")
        
        return result
    
    def has_production_changes(self) -> bool:
        """Check if there are any production-level changes."""
        return len(self.production_changes) > 0
    
    def has_curve_changes(self) -> bool:
        """Check if there are any curve-level changes."""
        return len(self.curve_changes) > 0
    
    def has_matrix_changes(self) -> bool:
        """Check if there are any matrix-level changes."""
        return len(self.matrix_changes) > 0
    
    def get_change_levels(self) -> List[str]:
        """Return list of levels that have changes defined (deepest first)."""
        levels = []
        if self.has_production_changes():
            levels.append(LEVEL_PRODUCTION)
        if self.has_curve_changes():
            levels.append(LEVEL_CURVE)
        if self.has_matrix_changes():
            levels.append(LEVEL_MATRIX)
        return levels
    
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
        """Return a human-readable summary of changes at all levels."""
        summary = [f"\nTechnological Change: {self.name}"]
        if self.description:
            summary.append(f"Description: {self.description}")
        
        total_changes = len(self.matrix_changes) + len(self.production_changes) + len(self.curve_changes)
        summary.append(f"\nTotal changes: {total_changes}")
        summary.append(f"Change levels: {', '.join(self.get_change_levels()) or 'None'}")
        
        # Matrix-level changes
        if self.matrix_changes:
            summary.append(f"\n--- Level 1: Matrix Changes ({len(self.matrix_changes)}) ---")
            for i, change in enumerate(self.matrix_changes, 1):
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
        
        # Production-level changes (Level 3 - deepest)
        if self.production_changes:
            summary.append(f"\n--- Level 3: Production Changes ({len(self.production_changes)}) ---")
            for i, change in enumerate(self.production_changes, 1):
                prod_id = change['production_id']
                field = change['field']
                ctype = change['change_type']
                val = change['value']
                
                if field == 'production_inputs':
                    input_isic = change.get('input_isic', 'all')
                    summary.append(f"  {i}. Production [{prod_id}] input {input_isic}: {ctype} by {val}")
                elif field == 'production_added_values':
                    comp = change.get('va_component', 'all')
                    summary.append(f"  {i}. Production [{prod_id}] VA.{comp}: {ctype} by {val}")
                else:
                    summary.append(f"  {i}. Production [{prod_id}] {field}: {ctype} by {val}")
        
        # Curve-level changes (Level 2)
        if self.curve_changes:
            summary.append(f"\n--- Level 2: Curve Changes ({len(self.curve_changes)}) ---")
            for i, change in enumerate(self.curve_changes, 1):
                isic = change['isic']
                action = change['action']
                
                if action == 'modify_tier':
                    tier = change['tier_index']
                    field = change['field']
                    ctype = change['change_type']
                    val = change['value']
                    summary.append(f"  {i}. Curve [{isic}] tier {tier} {field}: {ctype} by {val}")
                elif action == 'add_tier':
                    price = change['price']
                    cap = change['cap']
                    pos = change.get('position', -1)
                    summary.append(f"  {i}. Curve [{isic}] add tier at pos {pos}: price={price}, cap={cap}")
                elif action == 'remove_tier':
                    tier = change['tier_index']
                    summary.append(f"  {i}. Curve [{isic}] remove tier {tier}")
                elif action == 'scale_all_tiers':
                    field = change['field']
                    ctype = change['change_type']
                    val = change['value']
                    summary.append(f"  {i}. Curve [{isic}] all tiers {field}: {ctype} by {val}")
        
        # Capital requirements
        if self.capital_requirements:
            summary.append(f"\n--- Capital Investment Requirements ---")
            total_cost = self.get_total_capital_cost()
            summary.append(f"  Total capital cost: ${total_cost:,.2f}")
            summary.append(f"  Investment duration: {self.investment_duration} iteration(s)")
            for sector, amount in self.capital_requirements.items():
                summary.append(f"  Sector [{sector}]: ${amount:,.2f}")
        
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

