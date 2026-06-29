# Clean-data release notes

This document explains how to add the processed modelling table to the repository in a controlled way.

## Recommended file

Use:

```text
data/clean_data.zip
```

The zip should contain one file:

```text
clean_data.csv
```

Do not place a second copy of `clean_data.csv` in the repository root for public release. Root-level CSV copies are for local pipeline runs only and are ignored by `.gitignore`.

## Why zip the file?

A zipped processed table is easier to upload through GitHub, reduces repository size, and makes it explicit that the file is a curated release artifact rather than an active local working CSV.

## Minimum checks before commit

Run:

```bash
python scripts/check_clean_data_release.py
```

Confirm:

- the zip file exists;
- it contains `clean_data.csv`;
- the row count matches the intended processed table;
- the four target columns are present;
- the manifest SHA256 checksum is saved.

## Attribution

When releasing `clean_data.zip`, keep the ARC--MOF attribution in `docs/DATA_AVAILABILITY.md` and `CITATION.cff`. The processed table should be described as ARC--MOF-derived, not as raw ARC--MOF.

## What not to upload

Do not commit raw CIF archives, raw ARC--MOF tables, local cache folders, or trained model binaries unless there is a separate release decision documenting rights, size, and purpose.
