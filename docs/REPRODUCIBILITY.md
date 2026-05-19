# Reproducibility guide

## Environment setup

Using pip:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Windows PowerShell activation:

```powershell
.venv\Scripts\Activate.ps1
```

Using conda:

```bash
conda env create -f environment.yml
conda activate mof-failure-atlas
```

## Running the pipeline

Place required input CSV files in the repository root, then run:

```bash
python interpretable_failure_maps_pipeline.py --n_jobs 0
```

## Fixed settings in the current pipeline

| Setting | Value |
|---|---:|
| Number of train/test splits | 5 |
| Test fraction | 0.20 |
| Base random seed | 42 |
| Elite threshold | Top 10% of training target distribution |
| Minimum group size for reporting | 30 |
| Trust-category quantiles | 33rd and 66th percentiles |
| RF trees | 80 |
| RF max depth | 14 |
| HGB max iterations | 120 |
| HGB max depth | 6 |

## Elite-candidate misclassification

For each train/test split, the elite threshold is the 90th percentile of the training target distribution. A test structure is treated as elite if its true or predicted uptake is at or above this training-derived threshold. Elite-candidate misclassification is the disagreement between true and predicted elite/non-elite labels.

The pipeline also saves:

```text
elite_false_negative
elite_false_positive
elite_threshold_train
```

## Local trust categories

Trust categories are computed from:

- `mean_abs_error_across_models`
- `prediction_spread_std`

For each target-level disagreement table, the pipeline computes 33rd and 66th quantiles for error and disagreement:

- easy/stable: low error and low disagreement
- hard/unstable: high error and high disagreement
- hard/consistent: high error but not high disagreement
- ambiguous/model-sensitive: high disagreement but not high error
- intermediate: remaining cases

The implementation is in:

```text
calculate_local_trust_category()
```

## Output reproducibility

The main generated output folder is:

```text
failure_maps_outputs/
```

To rerun all model fits:

```bash
python interpretable_failure_maps_pipeline.py --force_model_fits
```

To regenerate merge and splits:

```bash
python interpretable_failure_maps_pipeline.py --force_merge --force_splits
```

## Important limitation

This repository does not redistribute the raw MOF source database files. Users must obtain them separately and comply with the original dataset licence/citation requirements.
