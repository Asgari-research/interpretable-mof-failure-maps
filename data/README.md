# Data folder

This folder is a placeholder for local data.

Raw MOF database/source files are not redistributed in this repository.

Expected local workflow:

```text
data/raw/        Optional local raw-data storage
data/interim/    Optional intermediate files
data/processed/  Optional processed local files
data/source_data/ Lightweight derived source-data tables, if redistribution is permitted
```

The current self-contained pipeline expects input CSV files in the repository root, next to `interpretable_failure_maps_pipeline.py`.
