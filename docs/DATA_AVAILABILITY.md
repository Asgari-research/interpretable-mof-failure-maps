# Data availability

The raw MOF database/source files used to construct the modelling cohort are **not redistributed** in this repository.

## Why raw data are not included

The input files are derived from external MOF database resources and associated tabular descriptor/cluster/topology files. To respect original data-provider licences, citation requirements, and file-size constraints, users must obtain the source files directly from the original database records.

## What this repository provides

This repository provides:

- a self-contained Python pipeline
- environment files
- documentation of expected inputs and outputs
- reproducibility notes
- optional lightweight derived source-data tables when redistribution is permitted

## Expected local input files

The current pipeline expects these files next to `interpretable_failure_maps_pipeline.py`:

```text
clean_data.csv
geo-clusters.csv
mc-clusters.csv
func-clusters.csv
flig-clusters.csv
all_topology_lists.csv
```

These files are intentionally ignored by Git.

## Generated outputs

Generated outputs are written under:

```text
failure_maps_outputs/
```

This folder is ignored by Git by default.

## Citation of source data

Users must cite the original MOF database/source-data publications and dataset records used to obtain the raw data. The associated manuscript and this repository should also be cited when using the code or derived outputs.
