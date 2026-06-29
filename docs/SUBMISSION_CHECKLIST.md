# Submission checklist

Use this checklist before linking the repository in the manuscript or Supporting Information.

## Repository text

- [ ] Main README title matches the manuscript title.
- [ ] CITATION.cff title matches the manuscript title.
- [ ] No superseded failure-atlas title remains from the earlier repository draft.
- [ ] No internal draft markers, tutorial wording, or unfinished notes remain.
- [ ] The repository says `descriptor trust atlas` or `local reliability`, not only `failure map`, unless referring to generated failure diagnostics.

## Data

- [ ] `data/clean_data.zip` exists if processed data are being released.
- [ ] The zip contains `clean_data.csv`.
- [ ] `python scripts/check_clean_data_release.py` has been run.
- [ ] `data/clean_data_manifest.txt` is committed if `data/clean_data.zip` is committed.
- [ ] Raw source files and CIF archives are not committed.

## Code and environment

- [ ] `requirements.txt` is one package per line.
- [ ] `environment.yml` is valid YAML.
- [ ] `.gitignore` does not ignore the intended `data/clean_data.zip`.
- [ ] Generated folders such as `failure_maps_outputs/` are ignored.

## Manuscript alignment

- [ ] README states the strict common cohort as 263,735 structures.
- [ ] README lists the four adsorption targets correctly.
- [ ] README and docs explain that printed SI tables are extracts when applicable.
- [ ] Data availability text cites ARC--MOF and does not imply ownership of raw third-party data.

## Final validation

Run:

```bash
python scripts/validate_repository_release.py
```

Commit only after the script passes.
