import torch
import numpy as np
import os
import sys

# Ensure the scripts directory is in the Python path to import mock_tensorus_utils
# This is a common way to handle imports for scripts not part of an installed package.
# Adjust if your project structure or execution environment handles this differently.
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

try:
    from mock_tensorus_utils import TensorStorage, TensorOps
except ImportError:
    print("Error: Could not import TensorStorage and TensorOps from mock_tensorus_utils.")
    print("Please ensure mock_tensorus_utils.py is in the same directory or accessible in PYTHONPATH.")
    sys.exit(1)


def generate_tensors():
    """
    Generates and stores various random numerical tensors using TensorStorage.
    Also demonstrates retrieving them and applying TensorOps functions.
    """

    # 1. Define output directory and ensure it exists
    output_dir = "data/synthetic/tensorus_demonstration_datasets/numerical/"
    os.makedirs(output_dir, exist_ok=True)
    print(f"Ensured output directory exists: {output_dir}")

    # 2. Define the output HDF5 file path
    output_h5_file_path = os.path.join(output_dir, "random_numerical_tensors.h5")
    print(f"Output HDF5 file path: {output_h5_file_path}")

    # 3. Initialize TensorStorage
    storage = TensorStorage(persistence_path=output_h5_file_path)
    print("TensorStorage initialized.")

    # 4. Define the dataset name
    dataset_name = "random_tensors"
    print(f"Using dataset name: {dataset_name}")

    # 5. Check if dataset group exists, if not, create it
    if not storage.dataset_exists(dataset_name):
        storage.create_dataset(dataset_name)
        print(f"Dataset '{dataset_name}' created in {output_h5_file_path}.")
    else:
        print(f"Dataset '{dataset_name}' already exists in {output_h5_file_path}.")

    # 6. Generate and store random tensors
    print("\nGenerating and storing tensors...")

    # Vector (1D)
    tensor_1d = torch.rand(100)
    metadata_1d = {"shape": "1D", "description": "Random vector", "dtype": str(tensor_1d.dtype)}
    id_1d = storage.insert(dataset_name, tensor_1d, metadata_1d)
    print(f"  Stored 1D tensor (vector) with ID: {id_1d}, Shape: {tensor_1d.shape}")

    # Matrix (2D)
    tensor_2d = torch.rand(50, 50)
    metadata_2d = {"shape": "2D", "description": "Random matrix", "dtype": str(tensor_2d.dtype)}
    id_2d = storage.insert(dataset_name, tensor_2d, metadata_2d)
    print(f"  Stored 2D tensor (matrix) with ID: {id_2d}, Shape: {tensor_2d.shape}")

    # Cube (3D)
    tensor_3d = torch.rand(20, 20, 20)
    metadata_3d = {"shape": "3D", "description": "Random cube", "dtype": str(tensor_3d.dtype)}
    id_3d = storage.insert(dataset_name, tensor_3d, metadata_3d)
    print(f"  Stored 3D tensor (cube) with ID: {id_3d}, Shape: {tensor_3d.shape}")

    # Hypercube (4D)
    tensor_4d = torch.rand(10, 10, 10, 10)
    metadata_4d = {"shape": "4D", "description": "Random hypercube", "dtype": str(tensor_4d.dtype)}
    id_4d = storage.insert(dataset_name, tensor_4d, metadata_4d)
    print(f"  Stored 4D tensor (hypercube) with ID: {id_4d}, Shape: {tensor_4d.shape}")

    print("All tensors generated and stored.")

    # 7. Retrieve and print information about stored tensors
    print("\n--- Verifying Stored Tensors and Demonstrating TensorOps ---")
    tensors_with_meta = storage.get_dataset_with_metadata(dataset_name)

    if not tensors_with_meta:
        print("No tensors found in the dataset. Something went wrong during storage.")
        return

    for i, (tensor_data, metadata) in enumerate(tensors_with_meta):
        print(f"\nTensor {i+1}:")
        print(f"  Metadata Shape: {metadata.get('shape', 'N/A')}")
        print(f"  Description: {metadata.get('description', 'N/A')}")
        print(f"  Actual Tensor Shape: {tensor_data.shape}")
        print(f"  Data Type: {metadata.get('dtype', str(tensor_data.dtype))}") # Or str(tensor_data.dtype)

        # Demonstrate a TensorOps function
        # Example: Calculate mean for all, and determinant for 2D tensors
        mean_val = TensorOps.mean(tensor_data)
        print(f"  TensorOps.mean(): {mean_val}")

        if tensor_data.ndim == 2 and tensor_data.shape[0] == tensor_data.shape[1]:
            # Ensure it's a square matrix for determinant
            try:
                # Convert to float32 or float64 if not already, as determinant usually expects float
                if not np.issubdtype(tensor_data.dtype, np.floating):
                    tensor_data_float = tensor_data.astype(np.float32)
                else:
                    tensor_data_float = tensor_data

                # For NumPy arrays, linalg.det works. For PyTorch, it's torch.linalg.det
                # TensorOps should handle this.
                det_val = TensorOps.matrix_determinant(tensor_data_float)
                print(f"  TensorOps.matrix_determinant(): {det_val}")
            except Exception as e:
                print(f"  Could not compute determinant: {e}")
        elif tensor_data.ndim == 1:
            std_val = TensorOps.std(tensor_data)
            print(f"  TensorOps.std(): {std_val}")


if __name__ == '__main__':
    print("Running script: generate_random_numerical_tensors.py")
    generate_tensors()
    print("\nScript finished.")

    # Optional: Add a small test to show how to query
    output_h5_file_path = "data/synthetic/tensorus_demonstration_datasets/numerical/random_numerical_tensors.h5"
    storage_for_query = TensorStorage(persistence_path=output_h5_file_path)
    dataset_name = "random_tensors"

    print("\n--- Example Query ---")
    print(f"Querying dataset '{dataset_name}' for tensors with shape '2D':")
    query_results = storage_for_query.query(dataset_name, lambda m: m.get('shape') == '2D')
    for i, (data, meta) in enumerate(query_results):
        print(f"  Match {i+1} - Description: {meta.get('description')}, Shape: {data.shape}")

    if not query_results:
        print("  No 2D tensors found by query (this might be unexpected if generation was successful).")
