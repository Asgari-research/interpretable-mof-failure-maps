# GitHub upload steps

Repository:

```text
https://github.com/Asgari-research/interpretable-mof-failure-maps
```

## 1. Make a safety branch

```bash
git clone https://github.com/Asgari-research/interpretable-mof-failure-maps.git
cd interpretable-mof-failure-maps
git checkout -b repo-cleanup-submission
```

## 2. Copy the replacement package

Copy the contents of the provided replacement zip into the cloned repository root. Allow files such as `README.md`, `CITATION.cff`, `.gitignore`, `docs/*.md`, and `data/README.md` to be overwritten.

Do not overwrite the main Python pipeline unless you have a separately verified formatted version. This cleanup package is focused on repository documentation, metadata, data policy, and release checks.

## 3. Add clean data

Place your processed data zip at:

```text
data/clean_data.zip
```

The zip should contain `clean_data.csv`.

Then run:

```bash
python scripts/check_clean_data_release.py
```

## 4. Validate the repository

```bash
python scripts/validate_repository_release.py
```

Fix any reported errors before committing.

## 5. Review Git status

```bash
git status --short
```

Expected changed files include documentation, metadata, scripts, and possibly:

```text
data/clean_data.zip
data/clean_data_manifest.txt
```

Do not commit root-level local files such as:

```text
clean_data.csv
geo-clusters.csv
mc-clusters.csv
func-clusters.csv
flig-clusters.csv
all_topology_lists.csv
failure_maps_outputs/
```

## 6. Commit

```bash
git add README.md CITATION.cff CHANGELOG.md .gitignore requirements.txt environment.yml pyproject.toml
git add docs data scripts results manuscript_assets supplementary_assets notebooks src
git commit -m "Clean repository documentation and data release notes"
```

If you are adding the processed table:

```bash
git add data/clean_data.zip data/clean_data_manifest.txt
git commit -m "Add processed clean data release artifact"
```

## 7. Push

```bash
git push origin repo-cleanup-submission
```

Open a pull request, or merge into `main` after checking the rendered GitHub pages.
