# Mapping Trust in MOF Adsorption Predictions

Reproducibility package for the manuscript:

**Mapping Trust in MOF Adsorption Predictions: Chemically Auditable Descriptor Atlases for Machine-Learning Screening**

This repository supports the descriptor-trust-atlas workflow used to evaluate local reliability in machine-learning predictions of metal--organic framework (MOF) adsorption. The project does not treat a single global score as sufficient evidence for screening decisions. It stores the code, documentation, data-access notes, and compact derived data needed to inspect where predictions are locally reliable, where they are model-sensitive, and where additional simulation or experimental validation is warranted.

## Scope

The repository is intended to support:

- rerunning the tabular benchmark when the required input tables are available locally;
- inspecting the processed modelling table, if `data/clean_data.zip` is supplied;
- tracing manuscript tables and figures to generated source-data files;
- documenting what is included, what is not included, and what original data sources must be cited.

It is not a redistribution of the full ARC--MOF database, raw CIF archives, or raw third-party source tables.

## Repository layout

```text
interpretable-mof-failure-maps/
├── README.md
├── CITATION.cff
├── LICENSE
├── CHANGELOG.md
├── requirements.txt
├── environment.yml
├── pyproject.toml
├── interpretable_failure_maps_pipeline.py
├── data/
│   ├── README.md
│   ├── clean_data.zip                 
│   ├── raw/README.md
│   └── source_data/README.md
├── docs/
│   ├── DATA_AVAILABILITY.md
│   ├── DATA_SCHEMA.md
│   ├── EXPECTED_INPUTS.md
│   ├── OUTPUTS.md
│   ├── REPRODUCIBILITY.md
│   ├── MANUSCRIPT_MAPPING.md
│   ├── CLEAN_DATA_RELEASE.md
│   ├── GITHUB_UPLOAD_STEPS.md
│   └── SUBMISSION_CHECKLIST.md
├── results/README.md
├── manuscript_assets/README.md
├── supplementary_assets/README.md
├── notebooks/README.md
└── src/interpretable_failure_maps/
    └── __init__.py
```

## Data policy

The raw MOF source database files are not committed here. The repository may include a processed table, preferably as:

```text
data/clean_data.zip
```

The zip file should contain one CSV file:

```text
clean_data.csv
```

This processed table is a derived, machine-learning-ready benchmark table for the present manuscript. Users should cite the original ARC--MOF dataset and paper when using it. See `docs/DATA_AVAILABILITY.md` and `docs/CLEAN_DATA_RELEASE.md`.

## Main adsorption targets

The manuscript analyses four adsorption targets, reported in mmol g^-1:

| Target column | Meaning |
|---|---|
| `uptake(mmol/g) CO2 at 0.015 bar` | CO2 uptake at 0.015 bar |
| `uptake(mmol/g) CO2 at 0.15 bar` | CO2 uptake at 0.15 bar |
| `uptake(mmol/g) methane at 5.8 bar` | CH4 uptake at 5.8 bar |
| `uptake(mmol/g) methane at 65 bar` | CH4 uptake at 65 bar |

The strict common modelling cohort contains 263,735 structures after removal of nine rows with missing core numerical descriptors.

## Model panel

The benchmark uses three tabular model families:

- Ridge regression;
- Random forest regression;
- Histogram gradient boosting regression.

The workflow reports global metrics, local domain errors, elite-candidate misclassification, novelty--error relationships, model disagreement, and local trust categories.

## Quick start

Create an environment:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

On Windows PowerShell:

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

Or with conda:

```bash
conda env create -f environment.yml
conda activate mof-trust-atlas
```

If you are using the processed table, place it at:

```text
data/clean_data.zip
```

For workflows that expect local CSV input in the repository root, extract or copy `clean_data.csv` locally before running the pipeline. Do not commit root-level working CSV files unless the data policy has been checked.

Run the main pipeline:

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

Generated outputs are written under:

```text
failure_maps_outputs/
```

This folder is ignored by Git.

## Citation

If you use this repository, cite:

1. the associated manuscript;
2. the original ARC--MOF data record and publication;
3. the archived repository release, if a Zenodo DOI is created.

See `CITATION.cff` for software-citation metadata.

## License

Code and documentation in this repository are released under the MIT License unless otherwise stated. Raw third-party source data are not redistributed by this license and remain subject to their original terms.
