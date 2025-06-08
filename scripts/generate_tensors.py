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

    # Keep Existing Tensors
    tensor_a = np.random.rand(100, 3)
    tensor_b = np.random.rand(50, 10, 3)

    # Add New Tensors
    # Scalar (0D tensor)
    scalar_data = np.array(42.0)

    # Vector (1D tensor)
    vector_data = np.random.rand(150)

    # Grayscale image (2D tensor)
    image_grayscale_data = np.random.rand(32, 32)

    # RGB image (3D tensor)
    image_rgb_data = np.random.rand(16, 16, 3)

    # Video frames (4D tensor - e.g., 10 frames, 8x8 grayscale)
    video_frames_data = np.random.rand(10, 8, 8, 1)

    # Simulation data (5D tensor)
    simulation_data = np.random.rand(5, 6, 6, 3, 2)

    # Save all tensors
    np.savez_compressed(
        output_dir / "demo_tensors.npz",
        tensor_a=tensor_a,
        tensor_b=tensor_b,
        scalar_data=scalar_data,
        vector_data=vector_data,
        image_grayscale_data=image_grayscale_data,
        image_rgb_data=image_rgb_data,
        video_frames_data=video_frames_data,
        simulation_data=simulation_data,
    )
    print(f"Generated demo_tensors.npz in {output_dir.resolve()}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate demo tensor dataset")
    parser.add_argument("output_dir", type=Path, help="Directory to write dataset")
    args = parser.parse_args()
    generate_tensors(args.output_dir)


if __name__ == "__main__":
    main()
