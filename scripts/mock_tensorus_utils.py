import h5py
import numpy as np
import torch
import uuid

class TensorStorage:
    """
    A class for storing and retrieving tensors with metadata in an HDF5 file.
    """
    def __init__(self, persistence_path, persistence=True):
        """
        Initializes the TensorStorage.

        Args:
            persistence_path (str): The path to the HDF5 file.
            persistence (bool, optional): Flag indicating if data should be persisted.
                                         Currently, this mock always persists to HDF5.
                                         Defaults to True.
        """
        self.persistence_path = persistence_path
        # The persistence flag is noted, but this mock implementation always persists to HDF5.

    def create_dataset(self, dataset_name):
        """
        Creates a dataset (group) in the HDF5 file if it doesn't already exist.

        Args:
            dataset_name (str): The name of the dataset (group) to create.
        """
        with h5py.File(self.persistence_path, 'a') as f:
            if dataset_name not in f:
                f.create_group(dataset_name)

    def dataset_exists(self, dataset_name):
        """
        Checks if a dataset (group) with the given name exists in the HDF5 file.

        Args:
            dataset_name (str): The name of the dataset (group) to check.

        Returns:
            bool: True if the dataset exists, False otherwise.
        """
        with h5py.File(self.persistence_path, 'a') as f: # 'a' mode creates file if not exists
            return dataset_name in f

    def insert(self, dataset_name, tensor, metadata):
        """
        Inserts a tensor and its metadata into the specified dataset (group).

        Args:
            dataset_name (str): The name of the dataset (group) to insert into.
            tensor (torch.Tensor or np.ndarray): The tensor to store.
            metadata (dict): A dictionary of metadata associated with the tensor.

        Returns:
            str: The unique ID assigned to the stored tensor.
        """
        if not self.dataset_exists(dataset_name):
            self.create_dataset(dataset_name)

        if isinstance(tensor, torch.Tensor):
            tensor_np = tensor.cpu().numpy()
        elif isinstance(tensor, np.ndarray):
            tensor_np = tensor
        else:
            raise TypeError("Tensor must be a PyTorch Tensor or a NumPy ndarray.")

        tensor_id = uuid.uuid4().hex

        with h5py.File(self.persistence_path, 'a') as f:
            dataset_path = f"{dataset_name}/{tensor_id}"
            dset = f[dataset_name].create_dataset(tensor_id, data=tensor_np)
            for key, value in metadata.items():
                # Ensure metadata values are of supported types for HDF5 attributes
                if isinstance(value, (str, int, float, np.ndarray, list)):
                    if isinstance(value, list): # h5py doesn't directly support lists of strings well
                        value = np.array(value, dtype='S') # Convert list of strings to array of bytes
                    dset.attrs[key] = value
                elif value is None:
                    dset.attrs[key] = 'None' # Store None as string
                else:
                    # Attempt to convert other types to string
                    try:
                        dset.attrs[key] = str(value)
                    except TypeError:
                        print(f"Warning: Could not store metadata value for key '{key}' of type {type(value)}. Skipping.")


        return tensor_id

    def get_dataset_with_metadata(self, dataset_name):
        """
        Retrieves all tensors and their metadata from the specified dataset (group).

        Args:
            dataset_name (str): The name of the dataset (group) to retrieve from.

        Returns:
            list: A list of tuples, where each tuple is (tensor_data, metadata).
                  Returns an empty list if the dataset does not exist.
        """
        if not self.dataset_exists(dataset_name):
            return []

        results = []
        with h5py.File(self.persistence_path, 'r') as f:
            if dataset_name in f:
                group = f[dataset_name]
                for tensor_id in group.keys():
                    h5_dataset = group[tensor_id]
                    tensor_data = h5_dataset[:]
                    metadata = dict(h5_dataset.attrs)
                    # Convert 'None' string back to None
                    for k, v in metadata.items():
                        if isinstance(v, str) and v == 'None':
                            metadata[k] = None
                        elif isinstance(v, np.bytes_): # Handle byte strings from S type
                            metadata[k] = v.decode('utf-8', errors='replace')
                        elif isinstance(v, np.ndarray) and v.dtype.kind == 'S': # Handle array of byte strings
                             metadata[k] = [s.decode('utf-8', errors='replace') for s in v]

                    results.append((tensor_data, metadata))
        return results

    def query(self, dataset_name, query_lambda):
        """
        Retrieves tensors and metadata from a dataset, filtered by a query function.

        Args:
            dataset_name (str): The name of the dataset (group) to query.
            query_lambda (function): A function that takes a metadata dictionary
                                     as input and returns True if the item matches.

        Returns:
            list: A list of tuples (tensor_data, metadata) for matching items.
        """
        all_items = self.get_dataset_with_metadata(dataset_name)
        if not all_items:
            return []

        matched_items = []
        for tensor_data, metadata in all_items:
            if query_lambda(metadata):
                matched_items.append((tensor_data, metadata))
        return matched_items


class TensorOps:
    """
    A class providing static methods for tensor operations.
    Uses torch for calculations if input is a torch.Tensor, otherwise numpy.
    """

    @staticmethod
    def _get_lib(tensor):
        """Determines whether to use torch or numpy based on tensor type."""
        return torch if isinstance(tensor, torch.Tensor) else np

    @staticmethod
    def mean(tensor, dim=None, keepdim=False):
        """
        Computes the mean of a tensor.

        Args:
            tensor (torch.Tensor or np.ndarray): The input tensor.
            dim (int or tuple of ints, optional): The dimension or dimensions to reduce.
            keepdim (bool, optional): Whether the output tensor has dim retained or not.

        Returns:
            torch.Tensor or np.ndarray: The mean of the tensor.
        """
        lib = TensorOps._get_lib(tensor)
        if lib == torch:
            if dim is None:
                # Global mean. torch.mean doesn't take keepdim if dim is None.
                return lib.mean(tensor)
            else:
                return lib.mean(tensor, dim=dim, keepdim=keepdim)
        else:  # numpy
            return lib.mean(tensor, axis=dim, keepdims=keepdim)

    @staticmethod
    def std(tensor, dim=None, keepdim=False):
        """
        Computes the standard deviation of a tensor.

        Args:
            tensor (torch.Tensor or np.ndarray): The input tensor.
            dim (int or tuple of ints, optional): The dimension or dimensions to reduce.
            keepdim (bool, optional): Whether the output tensor has dim retained or not.

        Returns:
            torch.Tensor or np.ndarray: The standard deviation of the tensor.
        """
        lib = TensorOps._get_lib(tensor)
        if lib == torch:
            # PyTorch's default is unbiased=True (sample std, ddof=1).
            # np.std default is ddof=0 (population std).
            # The prompt doesn't specify, so we use library defaults.
            if dim is None:
                # Global std. torch.std doesn't take keepdim if dim is None for global std.
                return lib.std(tensor)
            else:
                return lib.std(tensor, dim=dim, keepdim=keepdim)
        else:  # numpy
            return lib.std(tensor, axis=dim, keepdims=keepdim, ddof=0) # Specify ddof for clarity if needed

    @staticmethod
    def max(tensor, dim=None, keepdim=False):
        """
        Computes the maximum of a tensor.

        Args:
            tensor (torch.Tensor or np.ndarray): The input tensor.
            dim (int, optional): The dimension to reduce. (Note: np uses axis)
            keepdim (bool, optional): Whether the output tensor has dim retained or not. (Note: np uses keepdims)

        Returns:
            torch.Tensor or np.ndarray: The maximum of the tensor.
                                       If dim is specified and input is torch, returns (values, indices).
        """
        lib = TensorOps._get_lib(tensor)
        if lib == torch:
            if dim is None:
                return lib.max(tensor)  # Global max
            else:
                # torch.max returns (values, indices) when dim is specified
                return lib.max(tensor, dim=dim, keepdim=keepdim)
        else:  # numpy
            # np.max returns values, not (values, indices) like torch.max with dim
            return lib.max(tensor, axis=dim, keepdims=keepdim)

    @staticmethod
    def min(tensor, dim=None, keepdim=False):
        """
        Computes the minimum of a tensor.

        Args:
            tensor (torch.Tensor or np.ndarray): The input tensor.
            dim (int, optional): The dimension to reduce. (Note: np uses axis)
            keepdim (bool, optional): Whether the output tensor has dim retained or not. (Note: np uses keepdims)

        Returns:
            torch.Tensor or np.ndarray: The minimum of the tensor.
                                       If dim is specified and input is torch, returns (values, indices).
        """
        lib = TensorOps._get_lib(tensor)
        if lib == torch:
            if dim is None:
                return lib.min(tensor)  # Global min
            else:
                # torch.min returns (values, indices) when dim is specified
                return lib.min(tensor, dim=dim, keepdim=keepdim)
        else:  # numpy
            # np.min returns values, not (values, indices) like torch.min with dim
            return lib.min(tensor, axis=dim, keepdims=keepdim)

    @staticmethod
    def matrix_eigendecomposition(matrix):
        """
        Computes the eigendecomposition of a matrix.

        Args:
            matrix (torch.Tensor or np.ndarray): The input square matrix.

        Returns:
            tuple: (eigenvalues, eigenvectors)
                   For torch, eigenvalues are real for symmetric matrices, complex otherwise.
                   torch.linalg.eig returns complex eigenvalues and eigenvectors.
                   np.linalg.eig returns complex eigenvalues and eigenvectors.
        """
        lib = TensorOps._get_lib(matrix)
        if lib == torch:
            return lib.linalg.eig(matrix)
        else: # numpy
            return np.linalg.eig(matrix)

    @staticmethod
    def matrix_determinant(matrix):
        """
        Computes the determinant of a square matrix.

        Args:
            matrix (torch.Tensor or np.ndarray): The input square matrix.

        Returns:
            torch.Tensor or np.ndarray: The determinant of the matrix.
        """
        lib = TensorOps._get_lib(matrix)
        return lib.linalg.det(matrix)

    @staticmethod
    def matrix_inverse(matrix):
        """
        Computes the inverse of a square matrix.

        Args:
            matrix (torch.Tensor or np.ndarray): The input square matrix.

        Returns:
            torch.Tensor or np.ndarray: The inverse of the matrix.
        """
        lib = TensorOps._get_lib(matrix)
        return lib.linalg.inv(matrix)

    @staticmethod
    def correlation(tensor_2d):
        """
        Computes the correlation matrix of a 2D tensor.
        Assumes input is 2D (e.g., features x samples or samples x features).

        Args:
            tensor_2d (torch.Tensor or np.ndarray): The input 2D tensor.

        Returns:
            torch.Tensor or np.ndarray: The correlation matrix.
        """
        lib = TensorOps._get_lib(tensor_2d)
        if tensor_2d.ndim != 2:
            raise ValueError("Input tensor must be 2D for correlation.")

        if lib == torch:
            # PyTorch's corrcoef handles 2D tensors.
            # By default, rows are variables and columns are observations.
            return lib.corrcoef(tensor_2d)
        else: # numpy
            # np.corrcoef by default assumes rows are variables, columns are observations.
            return np.corrcoef(tensor_2d)

    @staticmethod
    def convolve_2d(image_batch, kernel_batch, padding=0, stride=1):
        """
        Performs a 2D convolution. Handles 4D inputs (batch, channels, H, W).

        Args:
            image_batch (torch.Tensor or np.ndarray): The input image batch (N, C_in, H_in, W_in).
            kernel_batch (torch.Tensor or np.ndarray): The kernel batch (N_out, C_in, kH, kW).
                                                     Or (C_out, C_in/groups, kH, kW) for torch.
            padding (int or str, optional): Padding added to both sides of the input. Defaults to 0.
            stride (int or tuple, optional): Stride of the convolution. Defaults to 1.

        Returns:
            torch.Tensor or np.ndarray: The result of the convolution.
        """
        # This operation is more naturally handled by PyTorch.
        # NumPy does not have a direct equivalent for batched 2D convolution with groups etc.
        # For simplicity, we'll primarily rely on PyTorch for this if available,
        # or raise a NotImplementedError for NumPy if direct conversion isn't straightforward.

        is_torch_input = isinstance(image_batch, torch.Tensor) and isinstance(kernel_batch, torch.Tensor)
        is_numpy_input = isinstance(image_batch, np.ndarray) and isinstance(kernel_batch, np.ndarray)

        if is_torch_input:
            # Ensure kernel is (out_channels, in_channels/groups, kH, kW)
            # If kernel_batch is (N_out, C_in, kH, kW), it should work if N_out is out_channels
            # and groups=1 implicitly.
            # torch.nn.functional.conv2d expects kernel to be (out_channels, in_channels/groups, kH, kW)
            return torch.nn.functional.conv2d(image_batch, kernel_batch, padding=padding, stride=stride)
        elif is_numpy_input:
            # For NumPy, we can convert to PyTorch tensors, perform convolution, and convert back.
            # This adds a dependency on PyTorch even for NumPy inputs for this specific function.
            # A pure NumPy implementation would be much more complex (e.g., using loops or stride_tricks).
            try:
                image_torch = torch.from_numpy(image_batch.astype(np.float32))
                kernel_torch = torch.from_numpy(kernel_batch.astype(np.float32))
                result_torch = torch.nn.functional.conv2d(image_torch, kernel_torch, padding=padding, stride=stride)
                return result_torch.numpy()
            except Exception as e:
                raise NotImplementedError(f"convolve_2d with NumPy inputs requires PyTorch and conversion failed, or is not directly implemented without it. Error: {e}")
        else:
            raise TypeError("Inputs for convolve_2d must be both PyTorch Tensors or both NumPy ndarrays.")

if __name__ == '__main__':
    # Example Usage (Optional: for testing the script directly)

    # --- TensorStorage Example ---
    storage_path = 'temp_tensor_storage.h5'
    storage = TensorStorage(persistence_path=storage_path)

    # Create a dataset
    dataset_name = "my_experiment_data"
    storage.create_dataset(dataset_name)
    print(f"Dataset '{dataset_name}' exists: {storage.dataset_exists(dataset_name)}")

    # Insert tensors
    tensor1 = torch.randn(3, 4)
    metadata1 = {'source': 'synthetic', 'type': 'float32', 'description': 'Random data 1', 'tags': ['tagA', 'tagB']}
    id1 = storage.insert(dataset_name, tensor1, metadata1)
    print(f"Inserted tensor 1 with ID: {id1}")

    tensor2 = np.array([[1, 2, 3], [4, 5, 6]], dtype=np.int32)
    metadata2 = {'source': 'manual', 'type': 'int32', 'description': 'Integer data', 'empty_list_test': []}
    id2 = storage.insert(dataset_name, tensor2, metadata2)
    print(f"Inserted tensor 2 with ID: {id2}")

    # Get all data from dataset
    all_data = storage.get_dataset_with_metadata(dataset_name)
    print(f"\nAll data in '{dataset_name}':")
    for i, (data, meta) in enumerate(all_data):
        print(f" Item {i+1} - Data shape: {data.shape}, Metadata: {meta}")

    # Query data
    print("\nQuerying for tensors with source 'synthetic':")
    query_results = storage.query(dataset_name, lambda m: m.get('source') == 'synthetic')
    for i, (data, meta) in enumerate(query_results):
        print(f" Match {i+1} - Data shape: {data.shape}, Metadata: {meta}")

    # --- TensorOps Example ---
    print("\n--- TensorOps Examples ---")
    np_array = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
    torch_tensor = torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])

    print(f"Mean of np_array: {TensorOps.mean(np_array)}")
    print(f"Mean of torch_tensor: {TensorOps.mean(torch_tensor)}")
    print(f"Std of np_array (axis=0): {TensorOps.std(np_array, dim=0)}")
    # For torch.std, dim is an int or tuple of ints
    print(f"Std of torch_tensor (dim=0): {TensorOps.std(torch_tensor, dim=0)}")


    # Max/Min
    print(f"Max of np_array: {TensorOps.max(np_array)}")
    # Torch max with dim returns (values, indices)
    # print(f"Max of torch_tensor (dim=1, keepdim=True): {TensorOps.max(torch_tensor, dim=1, keepdim=True)}")
    max_val_torch, max_idx_torch = TensorOps.max(torch_tensor, dim=1, keepdim=True)
    print(f"Max of torch_tensor (dim=1, keepdim=True): values={max_val_torch}, indices={max_idx_torch}")


    # Linear Algebra
    square_matrix_np = np.array([[2.0, 1.0], [1.0, 3.0]])
    square_matrix_torch = torch.tensor([[2.0, 1.0], [1.0, 3.0]])

    eigvals_np, eigvecs_np = TensorOps.matrix_eigendecomposition(square_matrix_np)
    print(f"Eigendecomposition of np_matrix: eigenvalues={eigvals_np}, eigenvectors=\n{eigvecs_np}")
    eigvals_torch, eigvecs_torch = TensorOps.matrix_eigendecomposition(square_matrix_torch)
    print(f"Eigendecomposition of torch_matrix: eigenvalues={eigvals_torch}, eigenvectors=\n{eigvecs_torch}")

    print(f"Determinant of np_matrix: {TensorOps.matrix_determinant(square_matrix_np)}")
    print(f"Determinant of torch_matrix: {TensorOps.matrix_determinant(square_matrix_torch)}")

    # Correlation
    data_for_corr_np = np.random.rand(10, 3) # 10 samples, 3 features
    print(f"Correlation of np data (features x samples):\n{TensorOps.correlation(data_for_corr_np.T)}") # Transpose to make rows features

    # Convolution (example requires float tensors)
    img_np = np.random.rand(1, 1, 10, 10).astype(np.float32) # Batch, InChannels, H, W
    kern_np = np.random.rand(1, 1, 3, 3).astype(np.float32) # OutChannels, InChannels, kH, kW
    conv_res_np = TensorOps.convolve_2d(img_np, kern_np, padding=1)
    print(f"Conv2D (NumPy inputs via PyTorch) result shape: {conv_res_np.shape}")

    img_torch = torch.rand(1, 1, 10, 10)
    kern_torch = torch.rand(1, 1, 3, 3) # OutChannels, InChannels/groups, kH, kW
    conv_res_torch = TensorOps.convolve_2d(img_torch, kern_torch, padding=1)
    print(f"Conv2D (PyTorch inputs) result shape: {conv_res_torch.shape}")

    # Clean up example HDF5 file
    import os
    if os.path.exists(storage_path):
        os.remove(storage_path)
        print(f"\nCleaned up {storage_path}")

    print("\nMock script created and basic examples tested (if run directly).")
