# Data availability

The repository separates processed manuscript data from raw third-party source data.

## Processed data included or intended for inclusion

The recommended public processed file is:

```text
data/clean_data.zip
```

This zip should contain:

```text
clean_data.csv
```

`clean_data.csv` is a derived, machine-learning-ready table for the descriptor-trust-atlas benchmark. It is provided to support transparency, reproducibility, and reuse of the reported analysis without requiring users to reconstruct the full working table from raw source files.

## Original data source

The underlying source data are derived from ARC--MOF. Users should cite the original ARC--MOF dataset record and associated publication when using `clean_data.csv` or outputs generated from it.

Original ARC--MOF data record:

```text
https://doi.org/10.5281/zenodo.6908728
```

Associated publication:

```bibtex
@article{burner2023arcmof,
  title   = {ARC--MOF: A Diverse Database of Metal--Organic Frameworks with DFT-Derived Partial Atomic Charges and Descriptors for Machine Learning},
  author  = {Burner, Jake and Schwiedrzik, Luca and Krykunov, Mykhaylo and Luo, Jun and Boyd, Peter G. and Woo, Tom K.},
  journal = {Chemistry of Materials},
  volume  = {35},
  number  = {3},
  pages   = {900--916},
  year    = {2023},
  doi     = {10.1021/acs.chemmater.2c02485}
}
```

## Files not redistributed

This repository does not redistribute the full raw ARC--MOF database, raw CIF archives, or raw adsorption/descriptors tables. The following are treated as local inputs or working files:

```text
geometric_properties.csv
post_comb_vsa-CO2.csv
methane.csv
geo-clusters.csv
mc-clusters.csv
func-clusters.csv
flig-clusters.csv
all_topology_lists.csv
```

## Generated outputs

Generated outputs are written under:

```text
failure_maps_outputs/
```

This folder is ignored by Git. Curated source-data tables may be copied into `data/source_data/` only after checking size, relevance, and redistribution rights.
