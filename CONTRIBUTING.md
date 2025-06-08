# Contributing Guidelines

Thank you for considering a contribution to this dataset repository. This project stores real world datasets alongside synthetic data generators. The following guidelines explain how to add new data or generator scripts and how to ensure a smooth review process.

## Adding a Real Dataset

1. Place the dataset inside `data/real_world/<dataset-name>`.
2. Include a `metadata.yaml` (or `metadata.json`) file next to the data files describing:
   - `source`: where the data originates from (link or citation)
   - `license`: terms of use for the dataset
   - `structure`: a short description of file organization and key columns or fields
   - `generation_steps`: how the data was acquired or processed
3. Use [Git LFS](https://git-lfs.com/) for any large files (e.g. `.csv`, `.zarr`, `.h5`). Ensure `.gitattributes` is updated if new file types are added.

## Adding a Synthetic Dataset or Generator

1. Place generated datasets in `data/synthetic/<dataset-name>`.
2. Copy the template at `data/synthetic/template_dataset/` as a starting point and fill in the placeholders.
3. If adding a generator script, put it under `scripts/` or a subdirectory.
4. Provide the same `metadata.yaml` with the required fields listed above describing how the data is produced.

## Commit Hygiene

- Keep commits focused and atomic; each commit should represent one logical change.
- Use clear commit messages that describe what was added or fixed.
- Do not commit data that should be tracked via Git LFS without first configuring LFS.

## Pull Request Expectations

- Open a pull request against the `main` branch once your changes are ready.
- Ensure the PR includes the dataset files, metadata, and any generator scripts.
- Be prepared to answer questions about the data source and licensing.
- Another contributor will review the PR for completeness and adherence to these guidelines.

We appreciate your effort to improve this repository!
