#!/usr/bin/env python3
"""Check repository hygiene before a public GitHub release."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

REQUIRED_FILES = [
    "README.md",
    "CITATION.cff",
    "CHANGELOG.md",
    ".gitignore",
    "requirements.txt",
    "environment.yml",
    "pyproject.toml",
    "data/README.md",
    "docs/DATA_AVAILABILITY.md",
    "docs/DATA_SCHEMA.md",
    "docs/EXPECTED_INPUTS.md",
    "docs/OUTPUTS.md",
    "docs/REPRODUCIBILITY.md",
    "docs/MANUSCRIPT_MAPPING.md",
    "docs/CLEAN_DATA_RELEASE.md",
    "docs/GITHUB_UPLOAD_STEPS.md",
    "docs/SUBMISSION_CHECKLIST.md",
]

BANNED_TEXT = [
    "Chemically Auditable Failure Atlases for Trustworthy MOF Adsorption Machine Learning",
    "AI-generated",
    "TBD",
    "TODO",
    "not sure",
]

RAW_FILES_THAT_SHOULD_NOT_BE_PUBLIC = [
    "clean_data.csv",
    "geo-clusters.csv",
    "mc-clusters.csv",
    "func-clusters.csv",
    "flig-clusters.csv",
    "all_topology_lists.csv",
    "geometric_properties.csv",
    "post_comb_vsa-CO2.csv",
    "methane.csv",
]

TEXT_EXTENSIONS = {".md", ".txt", ".yml", ".yaml", ".toml", ".cff", ".py", ".gitignore"}


def read_text(path: Path) -> str:
    try:
        return path.read_text(encoding="utf-8")
    except UnicodeDecodeError:
        return path.read_text(encoding="utf-8", errors="replace")


def main() -> int:
    errors: list[str] = []
    warnings: list[str] = []

    for rel in REQUIRED_FILES:
        if not (ROOT / rel).exists():
            errors.append(f"Missing required file: {rel}")

    for rel in RAW_FILES_THAT_SHOULD_NOT_BE_PUBLIC:
        if (ROOT / rel).exists():
            errors.append(f"Root-level raw/local input file should not be committed: {rel}")

    if (ROOT / "data" / "clean_data.csv").exists():
        warnings.append("data/clean_data.csv exists. Prefer data/clean_data.zip for the public release.")

    clean_zip = ROOT / "data" / "clean_data.zip"
    manifest = ROOT / "data" / "clean_data_manifest.txt"
    if clean_zip.exists() and not manifest.exists():
        warnings.append("data/clean_data.zip exists but data/clean_data_manifest.txt is missing. Run scripts/check_clean_data_release.py.")

    text_files = []
    for path in ROOT.rglob("*"):
        if not path.is_file():
            continue
        if ".git" in path.parts or ".venv" in path.parts or "__pycache__" in path.parts:
            continue
        rel_path = path.relative_to(ROOT).as_posix()
        if rel_path in {"scripts/validate_repository_release.py", "docs/SUBMISSION_CHECKLIST.md"}:
            continue
        if path.name == ".gitignore" or path.suffix in TEXT_EXTENSIONS:
            text_files.append(path)

    for path in text_files:
        rel = path.relative_to(ROOT).as_posix()
        text = read_text(path)
        for banned in BANNED_TEXT:
            if banned in text:
                errors.append(f"Banned or outdated text found in {rel}: {banned}")

    readme = ROOT / "README.md"
    if readme.exists():
        text = read_text(readme)
        if "Mapping Trust in MOF Adsorption Predictions" not in text:
            errors.append("README.md does not contain the current manuscript title.")
        if "263,735" not in text:
            warnings.append("README.md does not mention the strict 263,735-structure cohort.")

    citation = ROOT / "CITATION.cff"
    if citation.exists():
        text = read_text(citation)
        if "Mapping Trust in MOF Adsorption Predictions" not in text:
            errors.append("CITATION.cff does not contain the current manuscript title.")

    if errors:
        print("Repository release check FAILED")
        print("\nErrors:")
        for item in errors:
            print(f"- {item}")
        if warnings:
            print("\nWarnings:")
            for item in warnings:
                print(f"- {item}")
        return 1

    print("Repository release check PASSED")
    if warnings:
        print("\nWarnings:")
        for item in warnings:
            print(f"- {item}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
