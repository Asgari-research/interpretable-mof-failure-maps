# Reproducibility guide

## Environment setup

Using pip:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Windows PowerShell:

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

Using conda:

```bash
conda env create -f environment.yml
conda activate mof-trust-atlas
```

## Clean-data verification

After placing `data/clean_data.zip`, run:

```bash
python scripts/check_clean_data_release.py
```

The script writes a manifest with the SHA256 checksum and basic table metadata.

## Pipeline run

After placing the required local inputs, run:

```bash
python interpretable_failure_maps_pipeline.py --n_jobs 0
```

Useful rerun flags:

```bash
python interpretable_failure_maps_pipeline.py --force_merge
python interpretable_failure_maps_pipeline.py --force_splits
python interpretable_failure_maps_pipeline.py --force_model_fits
python interpretable_failure_maps_pipeline.py --skip_figures
```

## Fixed manuscript settings

| Setting | Value |
|---|---:|
| Train/test splits | 5 |
| Test fraction | 0.20 |
| Base random seed | 42 |
| Elite threshold | Top 10% of the training target distribution |
| Minimum group size for reported group metrics | 30 |
| Trust-category quantiles | 33rd and 66th percentiles |
| Ridge alpha | 1.0 |
| RF trees | 80 |
| RF maximum depth | 14 |
| RF minimum leaf size | 5 |
| HGB maximum iterations | 120 |
| HGB maximum depth | 6 |

## Elite-candidate misclassification

For each train/test split, the elite threshold is the 90th percentile of the training target distribution. A held-out structure is labelled elite if its true or predicted uptake exceeds that training-derived threshold. Elite-candidate misclassification is the disagreement between true and predicted elite/non-elite labels.

## Local trust categories

Trust categories are computed from absolute prediction error and cross-model prediction spread. The categories are screening strata, not calibrated uncertainty classes:

- easy/stable: low error and low disagreement;
- hard/unstable: high error and high disagreement;
- hard/consistent: high error but not high disagreement;
- model-sensitive: high disagreement but not high error;
- intermediate: remaining cases.

## Important limitation

This repository does not redistribute full raw source databases or raw CIF archives. Users must obtain raw third-party data from the original source records and follow the associated licence and citation requirements.
