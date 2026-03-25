"""
ISIC code utilities
====================

Functions to parse and validate ISIC classification codes.
Extracted from the original Evaluators.py.
"""

import logging
from typing import Dict

from db.repositories.good_repo import GoodsDatabase

logger = logging.getLogger(__name__)


def evaluate_good_isic(isic: str) -> dict:
    """
    Parse an ISIC code string into its hierarchical components.

    Parameters
    ----------
    isic : str
        ISIC code in the format ``"A01_01_001"``  (section+division+group+class,
        subclass, non-fungible suffix).

    Returns
    -------
    dict  with keys: isic_section, isic_division, isic_group, isic_class,
          sub_class_a, sub_class_b, sub_class_c, sub_class_nf
    """
    try:
        isic_core, subclass, non_fungible = isic.split("_")
        isic_section = isic_core[0]
        isic_division = isic_core[1:3]
        isic_group = isic_core[3:4]
        isic_class = isic_core[4:]
        return {
            "isic_section": isic_section,
            "isic_division": isic_division,
            "isic_group": isic_group,
            "isic_class": isic_class,
            "sub_class_a": subclass[0] if len(subclass) > 0 else None,
            "sub_class_b": subclass[1] if len(subclass) > 1 else None,
            "sub_class_c": subclass[2] if len(subclass) > 2 else None,
            "sub_class_nf": non_fungible if non_fungible else None,
        }
    except ValueError as exc:
        logger.error(f"Invalid ISIC format '{isic}': {exc}")
        return {
            "isic_section": None,
            "isic_division": None,
            "isic_group": None,
            "isic_class": None,
            "sub_class_a": None,
            "sub_class_b": None,
            "sub_class_c": None,
            "sub_class_nf": None,
        }


def evaluate_goods_isic() -> None:
    """
    Parse and persist ISIC classification fields for all goods in the database.

    Reads every :class:`~db.repositories.good_repo.Good` record, parses its
    ``isic`` string via :func:`evaluate_good_isic`, and writes the results
    back to the database.
    """
    gdb = GoodsDatabase()
    all_goods = gdb.get_all_goods()
    for good in all_goods:
        result = evaluate_good_isic(good.isic)  # type: ignore[arg-type]
        gdb.update_good(int(good.id), **result)  # type: ignore[arg-type]
