# Chemically Auditable Failure Atlases for MOF Adsorption Machine Learning

This repository contains the reproducibility package for the manuscript:

**Chemically Auditable Failure Atlases for Trustworthy MOF Adsorption Machine Learning**

The project builds interpretable failure atlases for MOF adsorption machine learning. Instead of reporting only global benchmark metrics, the workflow converts prediction outputs into chemically auditable diagnostics of local error, rank instability, model disagreement, descriptor-space novelty, and recurring hard-domain motifs.

## What this repository contains

The current release contains a self-contained Python pipeline:

```text
interpretable_failure_maps_pipeline.py
```

The script reads tabular MOF adsorption/descriptors/cluster/topology CSV files, trains a compact model panel, and generates benchmark metrics, failure-atlas source tables, trust-map diagnostics, and manuscript/SI figure assets.

## What this repository does **not** contain

The raw MOF database/source tables are **not redistributed** in this repository.

Users must obtain the required source data from the original database provider(s), comply with the original licences, and cite the original dataset/data-paper records. This repository only provides code, documentation, and optionally lightweight derived source-data tables when redistribution is permitted.

## Manuscript analysis summary

The manuscript uses a strict common cohort of **263,735 MOF structures** and four adsorption targets:

- CO2 uptake at 0.015 bar
- CO2 uptake at 0.15 bar
- CH4 uptake at 5.8 bar
- CH4 uptake at 65 bar

Three tabular model families are evaluated:

- Ridge regression
- Random forest regression
- Histogram gradient boosting regression

The workflow generates:

- global benchmark summaries
- split-level metrics
- per-sample prediction diagnostics
- domain-resolved failure maps
- hard-domain leaderboards
- novelty--error relationships
- model-disagreement and local-trust maps
- manuscript and Supporting Information source-data tables

## Required input files

The current self-contained script expects the following CSV files in the repository root, next to `interpretable_failure_maps_pipeline.py`:

```text
clean_data.csv
geo-clusters.csv
mc-clusters.csv
func-clusters.csv
flig-clusters.csv
all_topology_lists.csv
```

These files are ignored by `.gitignore` and should not be committed unless redistribution is explicitly permitted.

See [`docs/EXPECTED_INPUTS.md`](docs/EXPECTED_INPUTS.md) and [`docs/DATA_SCHEMA.md`](docs/DATA_SCHEMA.md).

## Quick start

### 1. Clone the repository

```bash
git clone https://github.com/Asgari-research/interpretable-mof-failure-maps.git
cd interpretable-mof-failure-maps
```

### 2. Create the environment

Using pip:

```bash
python -m venv .venv
source .venv/bin/activate  # Windows PowerShell: .venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

or using conda/mamba:

```bash
conda env create -f environment.yml
conda activate mof-failure-atlas
```

### 3. Add local input data

Place the required local CSV files in the repository root:

```text
clean_data.csv
geo-clusters.csv
mc-clusters.csv
func-clusters.csv
flig-clusters.csv
all_topology_lists.csv
```

Do **not** commit these files.

### 4. Run the pipeline

```bash
python interpretable_failure_maps_pipeline.py --n_jobs 0
```

Useful options:

```bash
python interpretable_failure_maps_pipeline.py --n_jobs 1
python interpretable_failure_maps_pipeline.py --force_merge
python interpretable_failure_maps_pipeline.py --force_splits
python interpretable_failure_maps_pipeline.py --force_model_fits
python interpretable_failure_maps_pipeline.py --skip_figures
```

## Main output folder

The script writes outputs to:

```text
failure_maps_outputs/
```

This generated output folder is ignored by Git by default.

Important generated subfolders include:

```text
failure_maps_outputs/results/metrics/
failure_maps_outputs/results/predictions/
failure_maps_outputs/results/tables/
failure_maps_outputs/results/figure_numeric_data/
failure_maps_outputs/manuscript_assets/figures/
failure_maps_outputs/supplementary_assets/figures/
failure_maps_outputs/final_exports/
```

See [`docs/OUTPUTS.md`](docs/OUTPUTS.md).

## Reproducibility notes

Key fixed settings in the current pipeline:

- 5 random train/test splits
- 20% held-out test fraction
- base random seed = 42
- elite-candidate threshold = top 10% of the training target distribution
- minimum group size for reported group metrics = 30
- local trust categories from 33rd/66th quantiles of mean absolute error and cross-model prediction spread

Full details are in [`docs/REPRODUCIBILITY.md`](docs/REPRODUCIBILITY.md).

## Repository layout

```text
docs/                   Documentation for data access, input schema, reproducibility, and outputs
data/                   Placeholder local-data folders; raw data are not committed
results/                Placeholder for generated/curated outputs; generated files are ignored
manuscript_assets/      Placeholder for manuscript figure/table assets
supplementary_assets/   Placeholder for SI figure/table assets
src/                    Package namespace placeholder for future modularization
interpretable_failure_maps_pipeline.py  Current self-contained pipeline
```

## Citation

If you use this repository, please cite:

1. The associated manuscript, once available.
2. The original MOF database/source-data publications and dataset records used to obtain the raw data.
3. This GitHub repository or archived release DOI, if a Zenodo release is created.

See [`CITATION.cff`](CITATION.cff).

## License

Code in this repository is released under the MIT License. Raw third-party data are not redistributed and remain subject to their original licences.
