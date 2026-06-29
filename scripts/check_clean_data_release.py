#!/usr/bin/env python3
"""Validate the processed clean-data release artifact.

Expected file:
    data/clean_data.zip

Expected content:
    clean_data.csv

The script prints basic metadata and writes:
    data/clean_data_manifest.txt
"""

from __future__ import annotations

import hashlib
import io
import sys
import zipfile
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
ZIP_PATH = ROOT / "data" / "clean_data.zip"
MANIFEST_PATH = ROOT / "data" / "clean_data_manifest.txt"
EXPECTED_CSV = "clean_data.csv"
EXPECTED_TARGETS = [
    "uptake(mmol/g) CO2 at 0.015 bar",
    "uptake(mmol/g) CO2 at 0.15 bar",
    "uptake(mmol/g) methane at 5.8 bar",
    "uptake(mmol/g) methane at 65 bar",
]


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> int:
    if not ZIP_PATH.exists():
        print(f"ERROR: Missing {ZIP_PATH.relative_to(ROOT)}")
        print("Place the processed table at data/clean_data.zip before running this check.")
        return 1

    checksum = sha256_file(ZIP_PATH)
    size_bytes = ZIP_PATH.stat().st_size

    with zipfile.ZipFile(ZIP_PATH, "r") as zf:
        names = [name for name in zf.namelist() if not name.endswith("/")]
        basenames = [Path(name).name for name in names]
        if EXPECTED_CSV not in basenames:
            print("ERROR: clean_data.zip must contain clean_data.csv")
            print("Found files:")
            for name in names:
                print(f"  - {name}")
            return 1

        csv_member = names[basenames.index(EXPECTED_CSV)]
        with zf.open(csv_member) as handle:
            raw = handle.read()
        df = pd.read_csv(io.BytesIO(raw), low_memory=False)

    missing_targets = [col for col in EXPECTED_TARGETS if col not in df.columns]
    row_count = len(df)
    column_count = len(df.columns)

    lines = [
        "clean_data.zip manifest",
        "=======================",
        f"file: data/clean_data.zip",
        f"sha256: {checksum}",
        f"size_bytes: {size_bytes}",
        f"csv_member: {csv_member}",
        f"rows: {row_count}",
        f"columns: {column_count}",
        "",
        "target_columns:",
    ]
    for col in EXPECTED_TARGETS:
        status = "present" if col in df.columns else "MISSING"
        n_missing = int(df[col].isna().sum()) if col in df.columns else "NA"
        lines.append(f"- {col}: {status}; missing={n_missing}")

    if "filename" in df.columns:
        lines.append(f"filename_unique: {df['filename'].nunique(dropna=True)}")
    if "id_norm" in df.columns:
        lines.append(f"id_norm_unique: {df['id_norm'].nunique(dropna=True)}")

    MANIFEST_PATH.write_text("\n".join(map(str, lines)) + "\n", encoding="utf-8")

    print("OK: clean_data.zip inspected")
    print(f"rows: {row_count:,}")
    print(f"columns: {column_count:,}")
    print(f"sha256: {checksum}")
    print(f"manifest: {MANIFEST_PATH.relative_to(ROOT)}")

    if missing_targets:
        print("ERROR: missing expected target columns:")
        for col in missing_targets:
            print(f"  - {col}")
        return 1

    if row_count not in {263735, 263744}:
        print("WARNING: row count is not 263,735 or 263,744. Verify that this is the intended release table.")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
