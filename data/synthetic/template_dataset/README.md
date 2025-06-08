# Sample Points Dataset

This folder contains a minimal example of a synthetic dataset. It demonstrates
how to bundle generated data along with a short dataset card and a generation
script.

## Dataset Card

- **Name**: Sample Points
- **Description**: A set of 100 random 2D points sampled uniformly from the
  square ``[-1, 1]``. The points are saved as both a CSV file and a compressed
  NumPy archive.
- **Source**: Generated locally using ``generation_scripts/generate_data.py``
  which relies on ``numpy.random``.
- **License**: MIT (see the repository ``LICENSE`` file).
- **Structure**:
    - ``points.csv`` – CSV file with header ``x,y`` containing the sampled
      coordinates.
    - ``points.npz`` – NumPy ``.npz`` archive with an array ``points`` of shape
      ``(100, 2)``.
- **Generation Steps**:
    1. From the repository root run:
       ``python data/synthetic/template_dataset/generation_scripts/generate_data.py data/synthetic/template_dataset``
    2. This creates or overwrites ``points.csv`` and ``points.npz`` in this
       folder.

## Usage

Describe how to load or use the generated data.
