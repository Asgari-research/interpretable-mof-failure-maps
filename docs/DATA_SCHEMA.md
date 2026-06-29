# Data schema

This document describes the expected processed-table schema for the descriptor-trust-atlas workflow.

## Identifier

The main processed table should include a structure identifier column:

```text
filename
```

The pipeline normalizes this identifier into:

```text
id_norm
```

Identifier normalization removes path components, strips whitespace, and removes `.cif` suffixes.

## Target columns

| Column | Manuscript meaning | Unit |
|---|---|---|
| `uptake(mmol/g) CO2 at 0.015 bar` | CO2 uptake at 0.015 bar | mmol g^-1 |
| `uptake(mmol/g) CO2 at 0.15 bar` | CO2 uptake at 0.15 bar | mmol g^-1 |
| `uptake(mmol/g) methane at 5.8 bar` | CH4 uptake at 5.8 bar | mmol g^-1 |
| `uptake(mmol/g) methane at 65 bar` | CH4 uptake at 65 bar | mmol g^-1 |

## Base numerical descriptor columns

The workflow uses available columns from this set:

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

## Engineered numerical descriptors

The pipeline computes or expects these derived quantities where possible:

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

## Categorical/domain descriptors

The workflow uses or constructs:

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

## Manuscript cohort

The manuscript reports a strict common cohort of 263,735 structures after removing nine rows with missing core numerical descriptors from the initial merged table.

If `data/clean_data.zip` is released, verify it with:

```bash
python scripts/check_clean_data_release.py
```
