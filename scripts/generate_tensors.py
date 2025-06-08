import argparse
from pathlib import Path

import numpy as np
import h5py


def generate_tensors(output_dir: Path) -> None:
    """Generate example tensors and save them to ``output_dir`` as ``demo_tensors.h5``.

    Parameters
    ----------
    output_dir : Path
        Directory where the tensor file will be written.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    output_file = output_dir / "demo_tensors.h5"

    # Keep Existing Tensors
    tensor_a = np.random.rand(100, 3)
    tensor_b = np.random.rand(50, 10, 3)

    # Add New Tensors
    # Scalar (0D tensor)
    scalar_data = np.array(42.0) # This is already a NumPy array

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

    # Save all tensors to HDF5
    with h5py.File(output_file, 'w') as hf:
        hf.create_dataset('tensor_a', data=tensor_a)
        hf.create_dataset('tensor_b', data=tensor_b)
        hf.create_dataset('scalar_data', data=scalar_data)
        hf.create_dataset('vector_data', data=vector_data)
        hf.create_dataset('image_grayscale_data', data=image_grayscale_data)
        hf.create_dataset('image_rgb_data', data=image_rgb_data)
        hf.create_dataset('video_frames_data', data=video_frames_data)
        hf.create_dataset('simulation_data', data=simulation_data)

    print(f"Generated {output_file.name} in {output_dir.resolve()}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate demo tensor dataset in HDF5 format")
    parser.add_argument("output_dir", type=Path, help="Directory to write dataset")
    args = parser.parse_args()
    generate_tensors(args.output_dir)


if __name__ == "__main__":
    main()
