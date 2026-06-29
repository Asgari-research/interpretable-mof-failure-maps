# Outputs

The main generated output folder is:

```text
failure_maps_outputs/
```

It is ignored by Git.

## Major output folders

```text
failure_maps_outputs/logs/
failure_maps_outputs/checkpoints/
failure_maps_outputs/data_processed/
failure_maps_outputs/split_definitions/
failure_maps_outputs/results/models/
failure_maps_outputs/results/predictions/
failure_maps_outputs/results/metrics/
failure_maps_outputs/results/tables/
failure_maps_outputs/results/figure_numeric_data/
failure_maps_outputs/manuscript_assets/figures/
failure_maps_outputs/supplementary_assets/figures/
failure_maps_outputs/final_exports/
```

## Important generated tables

Representative generated CSV files include:

```text
all_split_metrics.csv
table_main_benchmark_summary.csv
table_coverage_statistics.csv
table_group_level_error_summary.csv
table_hard_domain_rankings.csv
table_novelty_error_relationships.csv
table_agreement_disagreement_summary.csv
```

## Prediction-level outputs

Per-sample prediction files are generated under:

```text
failure_maps_outputs/results/predictions/
```

These files can be large and are not tracked by Git by default.

## Figure source data

Figure numeric source tables are written under:

```text
failure_maps_outputs/results/figure_numeric_data/
```

Curated figure source-data tables may be copied to `data/source_data/` after review.
