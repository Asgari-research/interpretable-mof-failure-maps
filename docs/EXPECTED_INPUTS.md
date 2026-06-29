# Expected inputs

The repository supports two practical modes.

## Mode A: Processed-table mode

Use this mode for the public repository release.

Expected file:

```text
data/clean_data.zip
```

The zip should contain:

```text
clean_data.csv
```

Some versions of the pipeline may expect `clean_data.csv` next to `interpretable_failure_maps_pipeline.py`. If so, extract the CSV locally for the run, but do not commit the extracted root-level CSV unless the data policy has been checked.

## Mode B: Full local regeneration mode

Use this mode only when rebuilding the processed table from local source files.

Expected local files may include:

```text
clean_data.csv
geo-clusters.csv
mc-clusters.csv
func-clusters.csv
flig-clusters.csv
all_topology_lists.csv
```

Optional raw/source files may include:

```text
geometric_properties.csv
post_comb_vsa-CO2.csv
methane.csv
```

These files are ignored by `.gitignore` because they may be large or subject to third-party redistribution terms.

## Identifier handling

The pipeline normalizes identifiers by taking the basename, removing `.cif`, and stripping whitespace before merging cluster/topology information.
