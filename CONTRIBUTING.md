# Contributing Guidelines

Thank you for considering a contribution to this dataset repository. This project stores real-world data sets alongside synthetic data generators. The sections below outline how to submit new datasets and maintain clean commits.

## Submitting a Dataset

1. Create a folder under `data/real_world/<dataset-name>` for real data or `data/synthetic/<dataset-name>` for synthetic data.
2. For synthetic datasets, start by copying the template directory located at `data/synthetic/template_dataset/` and replace the placeholders.
3. Add your dataset files and include a `metadata.yaml` (or `metadata.json`) file with the required fields listed below.
4. Use [Git LFS](https://git-lfs.com/) for any large files such as `.csv`, `.zarr`, or `.h5`. Update `.gitattributes` if you track new file types.

### Required Metadata Fields

Your metadata file must contain at minimum:

- `name`: human-readable dataset name
- `description`: short summary of what the dataset contains
- `source`: origin of the data or citation
- `license`: terms of use for the dataset
- `structure`: brief description of the directory layout or key columns
- `generation_steps`: how the data was collected or generated

## Commit Standards

- Keep commits atomic; each commit should represent one logical change.
- Write clear commit messages in the imperative mood (e.g. "Add new dataset").
- Do not commit data that should be tracked with Git LFS unless LFS is configured.
- Run any tests (`pytest`) before opening a PR.

## Pull Request Expectations

- Open a pull request against the `main` branch when your changes are ready.
- Ensure the PR includes the dataset files, metadata, and any generator scripts.
- Be prepared to answer questions about data provenance and licensing.
- Another contributor will review the PR for completeness and adherence to these guidelines.

We appreciate your effort to improve this repository!
