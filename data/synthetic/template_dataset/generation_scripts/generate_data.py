"""Skeleton generator script for creating synthetic data."""

from pathlib import Path

def generate(output_dir: Path) -> None:
    """Generate dataset files and save them to `output_dir`.

    Parameters
    ----------
    output_dir : Path
        Directory where the dataset files should be written.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    # TODO: add data generation logic here
    pass

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Generate synthetic dataset")
    parser.add_argument("output_dir", type=Path, help="Where to write generated data")
    args = parser.parse_args()
    generate(args.output_dir)
