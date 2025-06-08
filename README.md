# datasets

A repository for storing datasets and related resources used in experiments and tutorials. This project houses both real and synthetic data along with notebooks and helper scripts.

## Repository Layout

```
data/
├── real_world/    # real data collected from various sources
└── synthetic/     # generated datasets for testing and examples
notebooks/         # exploratory notebooks
scripts/           # data processing utilities
```

## Git LFS

Large dataset files should be managed with [Git LFS](https://git-lfs.com/).
Install Git LFS **before cloning** this repository so that large files are
fetched correctly. After installing it on your machine, run:

```bash
git lfs install
# Track large file types, for example
git lfs track "*.csv"
```

Commit the resulting `.gitattributes` file so that large data is stored efficiently.

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.
