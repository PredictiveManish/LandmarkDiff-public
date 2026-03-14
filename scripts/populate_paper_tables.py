#!/usr/bin/env python3
"""Auto-populate paper/main.tex tables from evaluation results.

Reads benchmark and ablation JSON results, then inserts the numeric
values into the LaTeX table placeholders (replacing "-- " entries).

Usage:
    python scripts/populate_paper_tables.py \
        --benchmark results/benchmark/benchmark.json \
        --ablation results/ablation/ablation_results.json \
        --tex paper/main.tex

    # Dry-run (prints changes without modifying file)
    python scripts/populate_paper_tables.py \
        --benchmark results/benchmark/benchmark.json \
        --tex paper/main.tex \
        --dry-run
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path


def load_json(path: str) -> dict | None:
    """Load JSON file, return None if not found."""
    p = Path(path)
    if not p.exists():
        print(f"Warning: {path} not found, skipping")
        return None
    with open(p) as f:
        return json.load(f)


def fmt_metric(value: float, metric_type: str) -> str:
    """Format a metric value for LaTeX."""
    if value is None or value != value:  # NaN check
        return "-- "
    if metric_type in ("ssim",):
        return f"{value:.3f}"
    elif metric_type in ("lpips",) or metric_type in ("nme",):
        return f"{value:.4f}"
    elif metric_type in ("fid",):
        return f"{value:.1f}"
    elif metric_type in ("arcface", "identity"):
        return f"{value:.3f}"
    return f"{value:.4f}"


def populate_table2(tex: str, benchmark: dict) -> str:
    """Populate Table 2 (main results) from benchmark data.

    Expected benchmark structure:
    {
        "methods": {
            "TPS Baseline": {"aggregate": {...}, "per_procedure": {...}},
            "Morphing": {...},
            "Phase A": {...},
            "Phase B (Ours)": {...},
        }
    }
    """
    methods = benchmark.get("methods", {})

    # Map method names to LaTeX row prefixes
    method_map = {
        "TPS Baseline": "TPS-only",
        "Morphing": "Morphing",
        "Phase A": "LandmarkDiff A",
        "Phase B (Ours)": "LandmarkDiff B",
    }

    # Map procedure names to LaTeX column names
    proc_map = {
        "rhinoplasty": "Rhinoplasty",
        "blepharoplasty": "Blepharoplasty",
        "rhytidectomy": "Rhytidectomy",
        "orthognathic": "Orthognathic",
    }

    changes = 0
    for method_key, method_data in methods.items():
        latex_method = method_map.get(method_key)
        if not latex_method:
            continue

        per_proc = method_data.get("per_procedure", {})
        method_data.get("aggregate", {})

        for proc_key, proc_name in proc_map.items():
            proc_data = per_proc.get(proc_key, {})
            if not proc_data:
                continue

            # Look for the row pattern in LaTeX and replace metrics
            # Table format: Method & Procedure & LPIPS & NME & SSIM & ArcFace\\
            # We search for the existing row and update the metrics
            ssim = proc_data.get("ssim")
            lpips = proc_data.get("lpips")
            nme = proc_data.get("nme")

            if ssim is not None:
                # Pattern: find rows with this method and procedure containing "--"
                # Replace "-- " values with actual numbers
                old_pat = re.compile(
                    rf"({re.escape(latex_method)}\s*&\s*{re.escape(proc_name)}\s*&\s*)"
                    r"([\d.]+|-- )\s*&\s*([\d.]+|-- )\s*&\s*([\d.]+|-- )\s*&\s*([\d.]+|-- )"
                )
                match = old_pat.search(tex)
                if match:
                    prefix = match.group(1)
                    old_lpips = match.group(2)
                    old_nme = match.group(3)
                    old_ssim = match.group(4)
                    old_arc = match.group(5)

                    new_lpips = fmt_metric(lpips, "lpips") if lpips is not None else old_lpips
                    new_nme = fmt_metric(nme, "nme") if nme is not None else old_nme
                    new_ssim = fmt_metric(ssim, "ssim") if ssim is not None else old_ssim
                    new_arc = old_arc  # Keep existing (no arcface from benchmark yet)

                    replacement = f"{prefix}{new_lpips} & {new_nme} & {new_ssim} & {new_arc}"
                    tex = tex[: match.start()] + replacement + tex[match.end() :]
                    changes += 1

    print(f"Table 2: {changes} rows updated")
    return tex


def populate_table3(tex: str, ablation: dict) -> str:
    """Populate Table 3 (ablation) from ablation results."""
    # Map ablation names to LaTeX identifiers
    ablation_map = {
        "Full TPS": "Full pipeline",
        "Full Pipeline (Ours)": "Full pipeline",
        "No Mask": "w/o mask",
        "Half Intensity": "50\\% intensity",
        "Double Intensity": "130\\% intensity",
        "Mesh Only": "Mesh only",
        "Canny Only": "Canny only",
        "No EMA": "w/o EMA",
        "No Curriculum": "w/o curriculum",
        "Phase A Only": "Phase A only",
    }

    changes = 0
    for abl_key, abl_data in ablation.items():
        agg = abl_data.get("aggregate", {})
        if not agg:
            continue

        ssim = agg.get("ssim_mean")
        lpips = agg.get("lpips_mean")
        nme = agg.get("nme_mean")

        latex_name = ablation_map.get(abl_key, abl_key)

        # Search for the row in the ablation table
        pat = re.compile(
            rf"({re.escape(latex_name)}\s*&\s*)"
            r"([\d.]+|-- )\s*&\s*([\d.]+|-- )\s*&\s*([\d.]+|-- )"
        )
        match = pat.search(tex)
        if match:
            prefix = match.group(1)
            new_ssim = fmt_metric(ssim, "ssim") if ssim is not None else match.group(2)
            new_lpips = fmt_metric(lpips, "lpips") if lpips is not None else match.group(3)
            new_nme = fmt_metric(nme, "nme") if nme is not None else match.group(4)

            replacement = f"{prefix}{new_ssim} & {new_lpips} & {new_nme}"
            tex = tex[: match.start()] + replacement + tex[match.end() :]
            changes += 1

    print(f"Table 3: {changes} rows updated")
    return tex


def populate_table4(tex: str, benchmark: dict) -> str:
    """Populate Table 4 (Fitzpatrick equity) from benchmark data."""
    methods = benchmark.get("methods", {})

    # Use the best method available (Phase B > Phase A > TPS)
    best = None
    for method_name in ["Phase B (Ours)", "Phase A", "TPS Baseline"]:
        if method_name in methods:
            best = methods[method_name]
            print(f"Table 4: using {method_name} for Fitzpatrick data")
            break

    if best is None:
        print("Table 4: no method data available")
        return tex

    fitz_data = best.get("per_fitzpatrick", {})
    changes = 0

    for fitz_type, data in fitz_data.items():
        ssim = data.get("ssim")
        lpips = data.get("lpips")
        nme = data.get("nme")
        data.get("n", 0)

        # Search for Fitzpatrick type row
        pat = re.compile(
            rf"(Type\s*{re.escape(fitz_type)}\s*.*?&\s*)"
            r"([\d.]+|-- )\s*&\s*([\d.]+|-- )\s*&\s*([\d.]+|-- )\s*&\s*([\d.]+|-- )"
        )
        match = pat.search(tex)
        if match:
            prefix = match.group(1)
            new_ssim = fmt_metric(ssim, "ssim") if ssim is not None else match.group(2)
            new_lpips = fmt_metric(lpips, "lpips") if lpips is not None else match.group(3)
            new_nme = fmt_metric(nme, "nme") if nme is not None else match.group(4)
            new_arc = match.group(5)  # Keep existing

            replacement = f"{prefix}{new_ssim} & {new_lpips} & {new_nme} & {new_arc}"
            tex = tex[: match.start()] + replacement + tex[match.end() :]
            changes += 1

    print(f"Table 4: {changes} rows updated")
    return tex


def main():
    parser = argparse.ArgumentParser(description="Populate paper tables from results")
    parser.add_argument(
        "--benchmark", default=None, help="Path to benchmark.json (for Tables 2, 4)"
    )
    parser.add_argument(
        "--ablation", default=None, help="Path to ablation_results.json (for Table 3)"
    )
    parser.add_argument("--tex", default="paper/main.tex", help="Path to LaTeX file")
    parser.add_argument("--dry-run", action="store_true", help="Print diff without modifying file")
    args = parser.parse_args()

    tex_path = Path(args.tex)
    if not tex_path.exists():
        print(f"Error: {args.tex} not found")
        sys.exit(1)

    original = tex_path.read_text()
    tex = original

    if args.benchmark:
        benchmark = load_json(args.benchmark)
        if benchmark:
            tex = populate_table2(tex, benchmark)
            tex = populate_table4(tex, benchmark)

    if args.ablation:
        ablation = load_json(args.ablation)
        if ablation:
            tex = populate_table3(tex, ablation)

    if tex == original:
        print("\nNo changes made to LaTeX file")
        return

    if args.dry_run:
        # Show diff
        import difflib

        diff = difflib.unified_diff(
            original.splitlines(keepends=True),
            tex.splitlines(keepends=True),
            fromfile=f"{args.tex} (original)",
            tofile=f"{args.tex} (updated)",
        )
        print("\n--- Changes (dry-run) ---")
        for line in diff:
            print(line, end="")
    else:
        tex_path.write_text(tex)
        print(f"\nUpdated {args.tex}")


if __name__ == "__main__":
    main()
