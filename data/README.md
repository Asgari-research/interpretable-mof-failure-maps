# Data

This folder documents the data files used by the repository. It distinguishes between processed manuscript data that may be shared here and raw third-party source files that should not be committed without checking the original licence and attribution terms.

## Recommended public data file

Add the processed modelling table as:

```text
data/clean_data.zip
```

The zip file should contain exactly one CSV file:

```text
clean_data.csv
```

This mirrors the release style used in the related holdout-validation repository: the repository can include a processed ARC--MOF-derived table for reproducibility while still avoiding redistribution of the full raw database.

## What `clean_data.csv` is

`clean_data.csv` is a derived, machine-learning-ready benchmark table prepared for this descriptor-trust-atlas study. It should contain adsorption targets, geometric descriptors, topology labels, and family/domain labels needed by the benchmark workflow.

It is not the original ARC--MOF database release and should not be described as raw data.

## What is not included

Do not commit these raw or local working files unless you have checked redistribution rights and file size:

```text
geometric_properties.csv
post_comb_vsa-CO2.csv
methane.csv
geo-clusters.csv
mc-clusters.csv
func-clusters.csv
flig-clusters.csv
all_topology_lists.csv
raw CIF archives
local generated output folders
trained model binaries
```

## After adding clean data

Run:

```bash
python scripts/check_clean_data_release.py
```

The script reports the row count, column count, target-column coverage, and SHA256 checksum, then writes:

```text
data/clean_data_manifest.txt
```

