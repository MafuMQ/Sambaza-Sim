"""
Verify that tech_changes.csv (new flat format) contains the same data
as tech_changes_old.csv (old JSON-params format).
"""
import csv
import json
from pathlib import Path
from pprint import pprint

DATA_DIR = Path("data/ex2")
OLD_FILE = DATA_DIR / "tech_changes_old.csv"
NEW_FILE = DATA_DIR / "tech_changes.csv"


# ── helpers ──────────────────────────────────────────────────────────────────

def load_old(path) -> dict:
    """
    Returns:  { example_id(int): { meta fields... , "params": [ {method, params} ] } }
    """
    result = {}
    with open(path, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            eid = int(row["example_id"])
            params_raw = row.get("tech_change_params", "").strip()
            params = json.loads(params_raw) if params_raw else []

            # normalise: flatten each param's inner "params" dict to the top level
            normalized = []
            for entry in params:
                flat = {"method": entry["method"]}
                flat.update(entry.get("params", {}))
                normalized.append(flat)

            result[eid] = {
                "tech_change_id":          row.get("tech_change_function_name", "").strip(),
                "change_type":             row["change_type"],
                "title":                   row["title"],
                "description":             row["description"],
                "final_demand":            row["final_demand"],
                "income_tax_rate_before":  row["income_tax_rate_before"],
                "income_tax_rate_after":   row["income_tax_rate_after"],
                "corporate_tax_rate_before": row["corporate_tax_rate_before"],
                "corporate_tax_rate_after":  row["corporate_tax_rate_after"],
                "income_tax_applies_to":   row["income_tax_applies_to"],
                "iterations":              row["iterations"],
                "use_multi_level":         row.get("use_multi_level", "").strip(),
                "ops": normalized,
            }
    return result


def _coerce(val: str):
    """Return int / float / bool / str as appropriate (empty string → None)."""
    if val == "":
        return None
    if val == "True":
        return True
    if val == "False":
        return False
    try:
        return int(val)
    except ValueError:
        pass
    try:
        return float(val)
    except ValueError:
        pass
    return val


OP_PARAM_COLS = [
    "sector_idx", "input_sector_idx", "production_id",
    "isic", "tier_index", "field", "change_type_param", "value",
    "efficiency_type", "va_component", "input_isic",
    "exclude_list", "position", "cap", "price",
    "va_components", "requirements", "investment_duration",
]


def _row_to_op(row: dict) -> dict:
    op = {"method": row["method"]}
    for col in OP_PARAM_COLS:
        v = row.get(col, "").strip()
        if v != "":
            # rename change_type_param → change_type to match old format
            key = "change_type" if col == "change_type_param" else col
            # special case: requirements is JSON
            if col == "requirements":
                op[key] = json.loads(v)
            elif col == "exclude_list":
                op[key] = json.loads(v) if v else []
            else:
                op[key] = _coerce(v)
    return op


def load_new(path) -> dict:
    """
    Returns same shape as load_old():
      { example_id(int): { meta fields... , "ops": [ flat-param dicts ] } }
    Groups multiple rows that share the same example_id.
    """
    result = {}
    with open(path, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            eid = int(row["example_id"])
            if eid not in result:
                result[eid] = {
                    "tech_change_id":          row["tech_change_id"].strip(),
                    "change_type":             row["change_type"],
                    "title":                   row["title"],
                    "description":             row["description"],
                    "final_demand":            row["final_demand"],
                    "income_tax_rate_before":  row["income_tax_rate_before"],
                    "income_tax_rate_after":   row["income_tax_rate_after"],
                    "corporate_tax_rate_before": row["corporate_tax_rate_before"],
                    "corporate_tax_rate_after":  row["corporate_tax_rate_after"],
                    "income_tax_applies_to":   row["income_tax_applies_to"],
                    "iterations":              row["iterations"],
                    "use_multi_level":         row.get("use_multi_level", "").strip(),
                    "ops": [],
                }
            result[eid]["ops"].append(_row_to_op(row))
    return result


# ── comparison ───────────────────────────────────────────────────────────────

def compare(old: dict, new: dict):
    issues = []

    old_ids = set(old.keys())
    new_ids = set(new.keys())

    if missing := old_ids - new_ids:
        issues.append(f"  IDs in OLD but missing from NEW: {sorted(missing)}")
    if extra := new_ids - old_ids:
        issues.append(f"  IDs in NEW but not in OLD: {sorted(extra)}")

    META_FIELDS = [
        "tech_change_id", "change_type", "title", "description",
        "final_demand", "income_tax_rate_before", "income_tax_rate_after",
        "corporate_tax_rate_before", "corporate_tax_rate_after",
        "income_tax_applies_to", "iterations", "use_multi_level",
    ]

    for eid in sorted(old_ids & new_ids):
        o, n = old[eid], new[eid]

        for field in META_FIELDS:
            ov, nv = o.get(field, ""), n.get(field, "")
            if str(ov) != str(nv):
                issues.append(
                    f"  example_id={eid} [{field}]  OLD={ov!r}  NEW={nv!r}"
                )

        o_ops, n_ops = o["ops"], n["ops"]
        if len(o_ops) != len(n_ops):
            issues.append(
                f"  example_id={eid}: op count mismatch  OLD={len(o_ops)}  NEW={len(n_ops)}"
            )
        else:
            for i, (oo, no) in enumerate(zip(o_ops, n_ops), 1):
                # exclude_inputs:[] in old is semantically equal to omitting it in new
                oo_norm = {k: v for k, v in oo.items() if not (k == "exclude_inputs" and v == [])}
                no_norm = {k: v for k, v in no.items() if not (k == "exclude_inputs" and v == [])}
                if oo_norm != no_norm:
                    issues.append(
                        f"  example_id={eid} op#{i} mismatch:\n"
                        f"    OLD: {oo}\n"
                        f"    NEW: {no}"
                    )

    return issues


# ── main ─────────────────────────────────────────────────────────────────────

def main():
    old = load_old(OLD_FILE)
    new = load_new(NEW_FILE)

    print("=" * 60)
    print(f"OLD has {len(old)} entries, NEW has {len(new)} entries")
    print("=" * 60)

    print("\n--- OLD (parsed) ---")
    for eid in sorted(old):
        print(f"\nexample_id={eid}  tech_change_id={old[eid]['tech_change_id']!r}")
        for op in old[eid]["ops"]:
            print(f"  {op}")

    print("\n--- NEW (parsed) ---")
    for eid in sorted(new):
        print(f"\nexample_id={eid}  tech_change_id={new[eid]['tech_change_id']!r}")
        for op in new[eid]["ops"]:
            print(f"  {op}")

    print("\n--- DIFF ---")
    issues = compare(old, new)
    if not issues:
        print("  No differences found — migration looks correct!")
    else:
        for issue in issues:
            print(issue)


if __name__ == "__main__":
    main()


