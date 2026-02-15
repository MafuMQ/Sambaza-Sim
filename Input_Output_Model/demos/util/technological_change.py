"""
Technological Change Module for Input-Output Analysis
======================================================

This module provides multi-level technological change modeling for Input-Output frameworks.
Changes can be applied at three levels:

**Level 1 - Matrix Level (High Level)**:
    Direct changes to the A matrix coefficients. Fastest, no rebuilding required.
    Use for: Quick what-if analysis, sensitivity testing.

**Level 2 - Productions Level (Micro Level)**:
    Changes to underlying production records (inputs, efficiencies, value added).
    Requires rebuilding the A matrix from modified production data.
    Use for: Firm-level technology adoption, process improvements.

**Level 3 - Curves Level (Market Level)**:
    Changes to supply curve tiers (capacities, prices, value added by tier).
    Requires rebuilding supply curve data for the solver.
    Use for: Capacity expansion, cost structure changes, market entry/exit.

Key Concepts:
-------------
1. Technical Coefficients (A matrix): Amount of input i needed per unit of output j
2. Productions: Individual production methods with input requirements and VA components
3. Supply Curves: Tiered supply with price/capacity steps per sector

Example Use Cases:
------------------
- Matrix Level: Reduce energy coefficients by 30% economy-wide
- Productions Level: A specific firm adopts new machinery (better efficiency)
- Curves Level: New capacity comes online at lower price tier

Usage:
------
    from Input_Output_Model.demos.util.technological_change import TechnologicalChange

    # === Level 1: Matrix-level change (current behavior) ===
    tech = TechnologicalChange(name="Energy Efficiency")
    tech.add_input_change(input_sector_idx=1, change_type="multiply", value=0.70)
    A_changed, VA_changed = tech.apply(A_baseline, VA_baseline)

    # === Level 2: Production-level change (requires rebuild) ===
    tech = TechnologicalChange(name="Factory Modernization")
    tech.add_production_change(
        production_id=5,  # Specific production method
        field="production_inputs",
        input_isic="A01",
        change_type="multiply",
        value=0.85  # 15% reduction in this input
    )
    tech.add_production_efficiency_change(
        production_id=5,
        efficiency_type="production_material_efficiency",
        change_type="add",
        value=10  # +10 efficiency points
    )
    # Rebuild matrices from modified productions
    A_new, VA_new, isic_map, modified_productions = tech.apply_to_productions(ptdb)

    # === Level 3: Curve-level change (requires rebuild) ===
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
    # Rebuild supply data from modified curves
    supply_data_new, modified_curves = tech.apply_to_curves(scdb, isic_map)
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
    
    1. Matrix Level: Direct changes to A matrix coefficients (fastest)
    2. Productions Level: Changes to production records (requires matrix rebuild)
    3. Curves Level: Changes to supply curve tiers (requires supply data rebuild)
    
    You can mix changes from different levels in the same TechnologicalChange object;
    they will be applied in order: productions -> curves -> matrix.
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
        
        # Level 2: Production-level changes (require matrix rebuild)
        self.production_changes = []
        
        # Level 3: Curve-level changes (require supply data rebuild)
        self.curve_changes = []
        
        # Backward compatibility alias
        self.changes = self.matrix_changes
    
    # ==========================================================================
    # Level 1: Matrix-Level Changes (High Level)
    # ==========================================================================
        
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
        
        self.matrix_changes.append({
            'level': LEVEL_MATRIX,
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
        self.matrix_changes.append({
            'level': LEVEL_MATRIX,
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
        self.matrix_changes.append({
            'level': LEVEL_MATRIX,
            'sector_idx': 'all',
            'input_sector_idx': input_sector_idx,
            'change_type': change_type,
            'value': value,
            'exclude_sectors': exclude_sectors or []
        })
    
    # ==========================================================================
    # Level 2: Production-Level Changes (Micro Level)
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
    # Level 3: Curve-Level Changes (Market Level)
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
    # Apply Methods
    # ==========================================================================
    
    def apply(self, A_matrix: np.ndarray, VA_vector: np.ndarray = None) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        """
        Apply matrix-level technological changes to an A matrix (and optionally VA vector).
        
        This method only applies LEVEL 1 (matrix-level) changes. For production-level
        or curve-level changes, use apply_to_productions() or apply_to_curves().
        
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
        
        logger.info(f"Applying matrix-level technological change: {self.name}")
        
        for change in self.matrix_changes:
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
    
    def apply_to_productions(self, ptdb, scdb=None, rebuild_matrix: bool = True,
                              loggingLevel=logging.WARNING) -> Dict[str, Any]:
        """
        Apply production-level changes to a ProductionsDatabase.
        
        This modifies production records and optionally rebuilds the IO matrix.
        Changes are applied in-memory; the database is NOT permanently modified.
        
        Parameters:
        -----------
        ptdb : ProductionsDatabase
            The productions database to read from
        scdb : SupplyCurveDatabase, optional
            Supply curve database (required if rebuild_matrix=True)
        rebuild_matrix : bool
            Whether to rebuild the A matrix after applying changes (default: True)
        loggingLevel : int
            Logging level for matrix build operations
        
        Returns:
        --------
        dict with:
            'productions': dict mapping production_id -> modified production data
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
        
        result = {'productions': modified_productions, 'A_matrix': None, 'VA_vector': None, 'isic_map': None}
        
        # Rebuild matrix if requested
        if rebuild_matrix:
            result.update(self._rebuild_matrix_from_productions(ptdb, scdb, modified_productions, loggingLevel))
        
        logger.info(f"Production-level changes applied: {len(modified_productions)} productions modified")
        return result
    
    def _rebuild_matrix_from_productions(self, ptdb, scdb, modified_productions: Dict, 
                                          loggingLevel=logging.WARNING) -> Dict[str, Any]:
        """
        Rebuild the A matrix and VA vector from production data, incorporating modifications.
        
        This creates a new matrix where modified productions are used instead of database values.
        """
        # Need to import here to avoid circular imports
        from Input_Output_Model.models.entities.SupplyCurve import SupplyCurveDatabase
        
        if scdb is None:
            scdb = SupplyCurveDatabase()
        
        # Get supply curves to determine matrix structure
        supply_curves = scdb.get_all_supply_curves()
        sorted_curves = sorted(supply_curves, key=lambda x: x.isic)
        
        n = len(sorted_curves)
        isic_map = {curve.isic: i for i, curve in enumerate(sorted_curves)}
        
        # Initialize matrices
        A_mon = np.zeros((n, n))
        VA_mon = np.zeros(n)
        
        # Fill matrix from productions (using modifications where applicable)
        for output_good in sorted_curves:
            col_idx = isic_map[output_good.isic]
            
            # Get productions for this good
            prods = ptdb.get_all_productions_by_good(int(output_good.id_number))
            if not prods:
                continue
            
            # Find the cheapest production (may be modified)
            best_prod = None
            best_price = float('inf')
            
            for p in prods:
                # Check if this production was modified
                if p.id in modified_productions:
                    prod_data = modified_productions[p.id]
                    price = prod_data['price']
                else:
                    price = float(p.price) if p.price else float('inf')
                
                if price < best_price:
                    best_price = price
                    best_prod = p
            
            if best_prod is None:
                continue
            
            # Use modified data if available, otherwise use original
            if best_prod.id in modified_productions:
                prod_data = modified_productions[best_prod.id]
                output_price = prod_data['price'] if prod_data['price'] > 0 else 1.0
                inputs = prod_data['production_inputs']
                total_va = prod_data['total_value_added']
            else:
                output_price = float(best_prod.price) if best_prod.price and float(best_prod.price) > 0 else 1.0
                inputs = best_prod.production_inputs or {}
                total_va = float(best_prod.total_value_added) if best_prod.total_value_added else 0.0
            
            # Fill intermediate input coefficients
            for input_isic, input_cost in inputs.items():
                if input_isic in isic_map:
                    row_idx = isic_map[input_isic]
                    A_mon[row_idx, col_idx] = float(input_cost) / output_price
            
            # Fill VA coefficient
            VA_mon[col_idx] = total_va / output_price
        
        return {'A_matrix': A_mon, 'VA_vector': VA_mon, 'isic_map': isic_map}
    
    def apply_to_curves(self, scdb, isic_map: Dict[str, int] = None) -> Dict[str, Any]:
        """
        Apply curve-level changes to supply curve data.
        
        This modifies supply curve tiers and returns the modified supply data
        for use with DynamicEquilibriumSolver.
        
        Parameters:
        -----------
        scdb : SupplyCurveDatabase
            The supply curve database to read from
        isic_map : dict, optional
            Mapping of ISIC codes to sector indices
        
        Returns:
        --------
        dict with:
            'supply_data': dict mapping ISIC -> modified tier list
            'modified_curves': dict of modified curve data
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
            # Parse price/cap tiers from JSON
            if curve.price and curve.total_inputs_cost:
                # Construct tiers from JSON arrays
                prices = curve.price if isinstance(curve.price, list) else [curve.price]
                caps = curve.total_inputs_cost if isinstance(curve.total_inputs_cost, list) else [curve.total_inputs_cost]
                va = curve.total_value_added if isinstance(curve.total_value_added, list) else [curve.total_value_added or 0]
                
                tiers = []
                for i in range(len(prices)):
                    tier = {
                        'price': float(prices[i]) if i < len(prices) else float(prices[-1]),
                        'cap': float(caps[i]) if i < len(caps) else -1,
                    }
                    # Add VA components if available
                    if curve.production_added_values:
                        va_data = curve.production_added_values
                        if isinstance(va_data, list) and i < len(va_data):
                            tier.update(va_data[i])
                        elif isinstance(va_data, dict):
                            tier.update(va_data)
                    tiers.append(tier)
                
                supply_data[curve.isic] = tiers
            elif curve.price:
                # Single tier
                supply_data[curve.isic] = [{
                    'price': float(curve.price) if not isinstance(curve.price, list) else float(curve.price[0]),
                    'cap': -1  # Unlimited
                }]
        
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
        return {'supply_data': supply_data, 'modified_curves': {isic: supply_data[isic] for isic in modified_curves}}
    
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
        """Return list of levels that have changes defined."""
        levels = []
        if self.has_matrix_changes():
            levels.append(LEVEL_MATRIX)
        if self.has_production_changes():
            levels.append(LEVEL_PRODUCTION)
        if self.has_curve_changes():
            levels.append(LEVEL_CURVE)
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
        
        # Production-level changes
        if self.production_changes:
            summary.append(f"\n--- Level 2: Production Changes ({len(self.production_changes)}) ---")
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
        
        # Curve-level changes
        if self.curve_changes:
            summary.append(f"\n--- Level 3: Curve Changes ({len(self.curve_changes)}) ---")
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
        
        return "\n".join(summary)
        
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
