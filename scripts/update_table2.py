"""Update Table 2 in paper/main.tex with ablation results.

Reads compositing ablation metrics from paper/ablation_results.json
and fills the placeholder rows in Table 2.

Usage:
    python scripts/update_table2.py [--dry-run]
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent


def load_ablation_results() -> dict:
    """Load ablation results JSON."""
    path = PROJECT_ROOT / "paper" / "ablation_results.json"
    if not path.exists():
        print(f"ERROR: {path} not found. Run ablation first.")
        sys.exit(1)
    return json.loads(path.read_text())


def fmt(val: dict | float, key: str = "mean") -> str:
    """Format a metric value for LaTeX."""
    v = val.get(key, 0) if isinstance(val, dict) else val
    return f"{v:.3f}"


def update_table2(tex: str, results: dict) -> str:
    """Replace placeholder rows in Table 2 with actual values."""

    # Map variant names to table row labels
    compositing_map = {
        "Direct output (no composite)": "direct_output",
        "Mask composite (no color match)": "mask_no_color",
        "Mask + LAB color match (full)": "mask_lab_full",
    }

    for row_label, variant_key in compositing_map.items():
        if variant_key not in results:
            continue

        m = results[variant_key]
        lpips = fmt(m.get("lpips", {}))
        nme = fmt(m.get("nme", {}))
        arcface = fmt(m.get("identity_sim", {}))

        # Find the row with placeholder dashes
        escaped_label = re.escape(row_label)
        pattern = rf"({escaped_label}\s*&\s*)--\s*&\s*--\s*&\s*--\s*\\\\"
        replacement = rf"\g<1>{lpips} & {nme} & {arcface} \\\\"
        tex, count = re.subn(pattern, replacement, tex)
        if count:
            print(f"  Updated: {row_label} -> LPIPS={lpips} NME={nme} ArcFace={arcface}")
        else:
            print(f"  WARN: Could not find placeholder row for '{row_label}'")

    # Bold the best value in each column for compositing section
    # (skip for now -- manual review preferred)

    return tex


def main():
    parser = argparse.ArgumentParser(description="Update Table 2 with ablation results")
    parser.add_argument("--dry-run", action="store_true", help="Print changes without writing")
    args = parser.parse_args()

    results = load_ablation_results()
    print(f"Loaded {len(results)} ablation variants")

    tex_path = PROJECT_ROOT / "paper" / "main.tex"
    tex = tex_path.read_text()
    updated = update_table2(tex, results)

    if updated == tex:
        print("No changes needed (already filled or no matching rows)")
        return

    if args.dry_run:
        print("\n[DRY RUN] Would write changes to", tex_path)
    else:
        tex_path.write_text(updated)
        print(f"\nUpdated {tex_path}")


if __name__ == "__main__":
    main()
