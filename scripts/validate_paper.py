#!/usr/bin/env python3
"""Validate paper/main.tex for completeness before submission.

Checks:
  1. No placeholder text (TODO, TBD, XXX, FIXME)
  2. All tables have data (no empty cells)
  3. All figures referenced and files exist
  4. All citations resolved (no ?)
  5. Word/page count within MICCAI limits
  6. Required sections present
  7. Abstract length

Usage:
    python scripts/validate_paper.py
    python scripts/validate_paper.py --tex paper/main.tex --strict
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_TEX = PROJECT_ROOT / "paper" / "main.tex"


def load_tex(tex_path: Path) -> str:
    """Load and return LaTeX source."""
    return tex_path.read_text()


def check_placeholders(tex: str) -> list[dict]:
    """Find placeholder text that should be replaced."""
    issues = []
    patterns = [
        (r"\bTODO\b", "TODO placeholder"),
        (r"\bTBD\b", "TBD placeholder"),
        (r"\bXXX\b", "XXX marker"),
        (r"\bFIXME\b", "FIXME marker"),
        (r"\bPLACEHOLDER\b", "PLACEHOLDER text"),
        (r"\\textcolor\{red\}", "Red-colored text (review marker)"),
        (r"\?\?\?", "??? placeholder"),
        (r"XX\.X", "XX.X numeric placeholder"),
        (r"0\.0000\b.*0\.0000", "Unfilled metric values"),
    ]
    for pattern, desc in patterns:
        for match in re.finditer(pattern, tex, re.IGNORECASE):
            line_num = tex[: match.start()].count("\n") + 1
            context = tex[max(0, match.start() - 30) : match.end() + 30].replace("\n", " ")
            issues.append(
                {
                    "type": "placeholder",
                    "severity": "error",
                    "line": line_num,
                    "message": f"{desc} at line {line_num}: ...{context}...",
                }
            )
    return issues


def check_tables(tex: str) -> list[dict]:
    """Check tables for empty or stub content."""
    issues = []

    # Find all tabular environments
    table_pattern = re.compile(
        r"\\begin\{tabular\}(.*?)\\end\{tabular\}",
        re.DOTALL,
    )

    for match in table_pattern.finditer(tex):
        table_content = match.group(1)
        line_num = tex[: match.start()].count("\n") + 1

        # Check for empty cells (& &) or stub values
        empty_cells = len(re.findall(r"&\s*&", table_content))
        dash_cells = len(re.findall(r"&\s*-\s*&", table_content))

        if empty_cells > 2:
            issues.append(
                {
                    "type": "table",
                    "severity": "error",
                    "line": line_num,
                    "message": f"Table at line {line_num} has {empty_cells} empty cells",
                }
            )

        if dash_cells > 3:
            issues.append(
                {
                    "type": "table",
                    "severity": "warning",
                    "line": line_num,
                    "message": f"Table at line {line_num} has {dash_cells} dash-only cells",
                }
            )

    return issues


def check_figures(tex: str) -> list[dict]:
    """Check all referenced figures exist."""
    issues = []

    # Find \includegraphics references
    fig_pattern = re.compile(r"\\includegraphics(?:\[.*?\])?\{([^}]+)\}")
    for match in fig_pattern.finditer(tex):
        fig_path = match.group(1)
        line_num = tex[: match.start()].count("\n") + 1

        # Try to resolve path
        full_path = PROJECT_ROOT / "paper" / fig_path
        if not full_path.exists():
            # Try without paper/ prefix
            full_path = PROJECT_ROOT / fig_path
        if not full_path.exists():
            # Try adding .png extension
            for ext in [".png", ".pdf", ".jpg", ".eps"]:
                candidate = PROJECT_ROOT / "paper" / (fig_path + ext)
                if candidate.exists():
                    full_path = candidate
                    break

        if not full_path.exists():
            issues.append(
                {
                    "type": "figure",
                    "severity": "error",
                    "line": line_num,
                    "message": f"Figure not found: {fig_path} (line {line_num})",
                }
            )

    return issues


def check_citations(tex: str) -> list[dict]:
    """Check for unresolved citations."""
    issues = []

    # Check for \cite{...} and find corresponding \bibitem or .bib entries
    cite_pattern = re.compile(r"\\cite[tp]?\{([^}]+)\}")
    citations = set()
    for match in cite_pattern.finditer(tex):
        keys = match.group(1).split(",")
        for key in keys:
            citations.add(key.strip())

    # Load .bib file if exists
    bib_entries = set()
    bib_path = PROJECT_ROOT / "paper" / "references.bib"
    if bib_path.exists():
        bib_content = bib_path.read_text()
        for match in re.finditer(r"@\w+\{(\w+),", bib_content):
            bib_entries.add(match.group(1))

    # Also check inline \bibitem
    for match in re.finditer(r"\\bibitem\{(\w+)\}", tex):
        bib_entries.add(match.group(1))

    # Find missing citations
    missing = citations - bib_entries
    if missing:
        for key in sorted(missing):
            issues.append(
                {
                    "type": "citation",
                    "severity": "error",
                    "line": 0,
                    "message": f"Unresolved citation: \\cite{{{key}}}",
                }
            )

    return issues


def check_sections(tex: str) -> list[dict]:
    """Check required sections are present."""
    issues = []

    required = [
        "abstract",
        "introduction",
        "related work",
        "method",
        "experiment",
        "result",
        "conclusion",
    ]

    tex_lower = tex.lower()
    for section in required:
        # Look for \section{...} or \begin{abstract}
        patterns = [
            rf"\\section\*?\{{[^}}]*{section}[^}}]*\}}",
            rf"\\begin\{{{section}\}}",
        ]
        found = False
        for p in patterns:
            if re.search(p, tex_lower):
                found = True
                break

        if not found:
            issues.append(
                {
                    "type": "section",
                    "severity": "warning",
                    "line": 0,
                    "message": f"Required section missing or differently named: '{section}'",
                }
            )

    return issues


def check_abstract(tex: str) -> list[dict]:
    """Check abstract length."""
    issues = []

    abstract_match = re.search(
        r"\\begin\{abstract\}(.*?)\\end\{abstract\}",
        tex,
        re.DOTALL,
    )
    if not abstract_match:
        issues.append(
            {
                "type": "abstract",
                "severity": "error",
                "line": 0,
                "message": "No abstract found",
            }
        )
        return issues

    abstract_text = abstract_match.group(1)
    # Remove LaTeX commands for word count
    clean = re.sub(r"\\[a-zA-Z]+\{[^}]*\}", "", abstract_text)
    clean = re.sub(r"[{}\\]", "", clean)
    words = len(clean.split())

    if words > 200:
        issues.append(
            {
                "type": "abstract",
                "severity": "warning",
                "line": 0,
                "message": f"Abstract may be too long: {words} words (MICCAI limit: ~150-200)",
            }
        )
    elif words < 50:
        issues.append(
            {
                "type": "abstract",
                "severity": "warning",
                "line": 0,
                "message": f"Abstract seems short: {words} words",
            }
        )

    return issues


def estimate_page_count(tex: str) -> int:
    """Rough page count estimate."""
    # Remove comments
    lines = [l for l in tex.split("\n") if not l.strip().startswith("%")]
    content = "\n".join(lines)
    # Remove LaTeX commands
    clean = re.sub(r"\\[a-zA-Z]+\{[^}]*\}", " ", content)
    clean = re.sub(r"[{}\\]", "", clean)
    words = len(clean.split())
    # Rough: ~500 words per page in 2-column
    return max(1, words // 500)


def validate_paper(tex_path: str, strict: bool = False) -> bool:
    """Run all validation checks on the paper."""
    path = Path(tex_path)
    if not path.exists():
        print(f"File not found: {tex_path}")
        return False

    tex = load_tex(path)

    print(f"{'=' * 60}")
    print(f"PAPER VALIDATION: {tex_path}")
    print(f"{'=' * 60}\n")

    all_issues = []
    all_issues.extend(check_placeholders(tex))
    all_issues.extend(check_tables(tex))
    all_issues.extend(check_figures(tex))
    all_issues.extend(check_citations(tex))
    all_issues.extend(check_sections(tex))
    all_issues.extend(check_abstract(tex))

    # Count by severity
    errors = [i for i in all_issues if i["severity"] == "error"]
    warnings = [i for i in all_issues if i["severity"] == "warning"]

    # Print issues
    if errors:
        print(f"ERRORS ({len(errors)}):")
        for issue in errors:
            print(f"  [ERROR] {issue['message']}")

    if warnings:
        print(f"\nWARNINGS ({len(warnings)}):")
        for issue in warnings:
            print(f"  [WARN]  {issue['message']}")

    # Stats
    pages = estimate_page_count(tex)
    n_citations = len(re.findall(r"\\cite[tp]?\{", tex))
    n_figures = len(re.findall(r"\\begin\{figure", tex))
    n_tables = len(re.findall(r"\\begin\{table", tex))

    print("\nSTATISTICS:")
    print(f"  Estimated pages: {pages}")
    print(f"  Citations: {n_citations}")
    print(f"  Figures: {n_figures}")
    print(f"  Tables: {n_tables}")

    # MICCAI limits
    if pages > 10:
        print(f"  WARNING: Estimated {pages} pages (MICCAI limit: 8 + references)")

    print(f"\n{'=' * 60}")
    if errors:
        print(f"VALIDATION FAILED: {len(errors)} errors, {len(warnings)} warnings")
    elif warnings:
        print(f"PASSED WITH WARNINGS: {len(warnings)} warnings")
    else:
        print("ALL CHECKS PASSED")
    print(f"{'=' * 60}")

    if strict:
        return len(errors) == 0 and len(warnings) == 0
    return len(errors) == 0


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Validate paper for submission")
    parser.add_argument("--tex", default=str(DEFAULT_TEX), help="LaTeX file to validate")
    parser.add_argument("--strict", action="store_true", help="Treat warnings as errors")
    args = parser.parse_args()
    success = validate_paper(args.tex, args.strict)
    sys.exit(0 if success else 1)
