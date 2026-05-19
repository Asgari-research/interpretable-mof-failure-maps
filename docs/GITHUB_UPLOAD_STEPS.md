# Manual GitHub upload steps

Repository URL:

```text
https://github.com/Asgari-research/interpretable-mof-failure-maps
```

## 1. Clone

```bash
git clone https://github.com/Asgari-research/interpretable-mof-failure-maps.git
cd interpretable-mof-failure-maps
```

## 2. Copy this package into the cloned folder

Copy all files from this prepared package into the cloned repository folder.

## 3. Check status

```bash
git status
```

Confirm that raw data files are not listed.

## 4. Add files

```bash
git add README.md LICENSE CITATION.cff .gitignore requirements.txt environment.yml
git add docs data results manuscript_assets supplementary_assets notebooks scripts src
git add interpretable_failure_maps_pipeline.py
```

## 5. Commit

```bash
git commit -m "Initialize interpretable MOF failure-map reproducibility package"
```

## 6. Push

If you are working directly on main:

```bash
git push origin main
```

If you prefer a branch:

```bash
git checkout -b repo-setup
git push origin repo-setup
```

Then open a pull request on GitHub.

## 7. Before pushing, verify dangerous files are not included

Run:

```bash
git status --short
```

Do not push:

```text
clean_data.csv
geo-clusters.csv
mc-clusters.csv
func-clusters.csv
flig-clusters.csv
all_topology_lists.csv
failure_maps_outputs/
*.joblib
*.pkl
*.parquet
```
