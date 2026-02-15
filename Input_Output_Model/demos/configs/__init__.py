"""
Demo Configuration Package
===========================

Contains example configurations for Input-Output model simulations:
- Tax policy examples (examples 1-6)
- Technological change examples (examples 7-12)
"""

from .tax_policy_examples import TAX_POLICY_EXAMPLES
from .tech_change_examples import TECH_CHANGE_EXAMPLES

__all__ = ['TAX_POLICY_EXAMPLES', 'TECH_CHANGE_EXAMPLES']
