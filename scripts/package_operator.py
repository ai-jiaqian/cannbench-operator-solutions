#!/usr/bin/env python3
"""Assemble a selected operator with the scaffold from its official source Job."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from shutil import copytree


ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("operator", help="operator_key from CATALOG.md")
    parser.add_argument("output", type=Path, help="new output directory")
    args = parser.parse_args()
    catalog = json.loads((ROOT / "results/current.json").read_text())
    row = next(
        (
            item
            for item in catalog["operators"]
            if item["operator_key"] == args.operator
        ),
        None,
    )
    if row is None:
        parser.error(f"unknown operator: {args.operator}")
    if args.output.exists():
        parser.error(f"output already exists: {args.output}")
    copytree(
        ROOT / "jobs" / row["job_id"],
        args.output,
        ignore=lambda _dir, names: {"provenance.json"} & set(names),
    )
    operator_root = ROOT / "operators" / args.operator / "csrc" / "ops"
    for op_dir in operator_root.iterdir():
        copytree(op_dir, args.output / "csrc" / "ops" / op_dir.name)
    print(f"Assembled {args.operator} from {row['job_id']} at {args.output}")


if __name__ == "__main__":
    main()
