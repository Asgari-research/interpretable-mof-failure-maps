# Expected input files

The current self-contained pipeline reads input CSV files from the same directory as:

```text
interpretable_failure_maps_pipeline.py
```

## Required files

| Filename | Role |
|---|---|
| `clean_data.csv` | Main merged adsorption/descriptors table |
| `geo-clusters.csv` | Geometry-family cluster assignments |
| `mc-clusters.csv` | Metal-cluster/family assignments |
| `func-clusters.csv` | Functional-family assignments |
| `flig-clusters.csv` | Linker-family assignments |
| `all_topology_lists.csv` | Topology labels and supplementary topology information |

## Important

These files are not included in the repository and are ignored by `.gitignore`.

Place them locally in the repository root before running:

```bash
python interpretable_failure_maps_pipeline.py --n_jobs 0
```

## Input identifier handling

The pipeline normalizes structure identifiers by:

- taking the basename of the file path,
- removing `.cif` suffixes,
- stripping whitespace.

This is used to merge cluster/topology tables with the main descriptor table.

## Main target columns expected by the pipeline

```text
uptake(mmol/g) CO2 at 0.015 bar
uptake(mmol/g) CO2 at 0.15 bar
uptake(mmol/g) methane at 5.8 bar
uptake(mmol/g) methane at 65 bar
```
