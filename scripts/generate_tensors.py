import argparse
from pathlib import Path

import numpy as np


def generate_tensors(output_dir: Path) -> None:
    """Generate example tensors and save them to ``output_dir`` as ``demo_tensors.npz``.

    Parameters
    ----------
    output_dir : Path
        Directory where the tensor file will be written.
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    tensor_a = np.random.rand(100, 3)
    tensor_b = np.random.rand(50, 10, 3)

    np.savez(output_dir / "demo_tensors.npz", tensor_a=tensor_a, tensor_b=tensor_b)


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate demo tensor dataset")
    parser.add_argument("output_dir", type=Path, help="Directory to write dataset")
    args = parser.parse_args()
    generate_tensors(args.output_dir)


if __name__ == "__main__":
    main()
