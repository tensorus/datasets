"""Generator for a simple 2D points dataset."""

from pathlib import Path
import csv
import numpy as np

def generate(output_dir: Path) -> None:
    """Generate dataset files and save them to `output_dir`.

    Parameters
    ----------
    output_dir : Path
        Directory where the dataset files should be written.
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    # Deterministic RNG for reproducibility
    rng = np.random.default_rng(0)

    # Sample 100 points uniformly from [-1, 1] x [-1, 1]
    points = rng.uniform(-1, 1, size=(100, 2))

    # Write CSV with header
    csv_file = output_dir / "points.csv"
    with csv_file.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["x", "y"])
        writer.writerows(points)

    # Also save as compressed NumPy archive
    npz_file = output_dir / "points.npz"
    np.savez(npz_file, points=points)

    print(f"Generated {csv_file.name} and {npz_file.name} in {output_dir.resolve()}")

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Generate synthetic dataset")
    parser.add_argument("output_dir", type=Path, help="Where to write generated data")
    args = parser.parse_args()
    generate(args.output_dir)
