# datasets

This repository stores datasets and related resources for experiments and tutorials. It contains curated real-world data, synthetic examples, Jupyter notebooks, and helper scripts.

## Repository Purpose

The goal is to provide a centralized location for datasets used in demos or research. Datasets are organized to keep real data separate from generated examples while also tracking notebooks and scripts that operate on them.

## Dataset Organization

```
data/
├── real_world/    # datasets collected from external sources
└── synthetic/     # generated data and templates
notebooks/         # exploration and analysis notebooks
scripts/           # utilities for generation or preprocessing
```
Executed notebooks (e.g., `demo_tensors.executed.ipynb`) are not tracked. Run the clean notebook to reproduce outputs.


## Git LFS

Large dataset files should be stored with [Git LFS](https://git-lfs.com/). Install Git LFS **before cloning** so binary files are fetched correctly:

```bash
git lfs install
```

Track new file types as needed. For example:

```bash
git lfs track "*.csv"
```

This repository tracks common dataset formats with LFS:

```bash
git lfs track "*.zarr" "*.h5" "*.npz" "*.safetensors"
```

Commit the resulting `.gitattributes` file along with your data. If you clone after enabling LFS, run `git lfs pull` to download the tracked content.

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

## Dashboard

A Streamlit dashboard is included for exploring the datasets interactively. Launch it with:

```bash
streamlit run scripts/dashboard.py
```
