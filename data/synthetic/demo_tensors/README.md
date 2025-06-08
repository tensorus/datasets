# Demo Tensors

This folder contains a small synthetic dataset of NumPy tensors generated with
`scripts/generate_tensors.py`.

## Files

- `demo_tensors.npz` – compressed archive containing two tensors:
  - `tensor_a`: shape `(100, 3)`
  - `tensor_b`: shape `(50, 10, 3)`

The tensors are randomly generated using `numpy.random.rand` and can be loaded
with `numpy.load`.
