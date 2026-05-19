# Data schema

This document describes the expected schema used by the current pipeline.

## Main identifier

The main table is expected to contain:

```text
filename
```

The pipeline creates a normalized identifier called:

```text
id_norm
```

## Target columns

The default targets are:

| Target column | Manuscript meaning |
|---|---|
| `uptake(mmol/g) CO2 at 0.015 bar` | CO2 uptake at 0.015 bar |
| `uptake(mmol/g) CO2 at 0.15 bar` | CO2 uptake at 0.15 bar |
| `uptake(mmol/g) methane at 5.8 bar` | CH4 uptake at 5.8 bar |
| `uptake(mmol/g) methane at 65 bar` | CH4 uptake at 65 bar |

All targets are treated as mmol g^-1.

## Base numerical features

The pipeline uses available columns from this set:

```text
Density
UC_volume
ASA
vASA
gASA
NASA
gNASA
vNASA
AVA
AVAf
AVAg
NAVA
NAVAf
NAVAg
POAVA
POAVAf
POAVAg
NPOAVA
NPOAVAf
NPOAVAg
Df
Di
Dif
```

## Engineered numerical features

The pipeline computes:

```text
lcd_pld_ratio
cavity_window_gap
sa_pv_ratio
vf_density_ratio
log_pld_plus1
log_lcd_plus1
avaf_x_density
pore_shape_ratio
```

## Categorical/domain features

The pipeline uses or creates:

```text
topology_group
topology_frequency_group
pore_regime
density_regime
geometry_family
metal_family
functional_family
linker_family
```

## Final cohort used in manuscript

The manuscript reports a final common cohort of 263,735 structures after removing nine rows with missing core numerical descriptors from the initial merged table.

## Notes

If your local raw column names differ, adapt the constants at the top of `interpretable_failure_maps_pipeline.py`.
