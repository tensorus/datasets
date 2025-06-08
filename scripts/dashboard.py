from __future__ import annotations

import h5py
import numpy as np
import pandas as pd
import streamlit as st
from pathlib import Path

DATA_DIR = Path(__file__).resolve().parents[1] / "data"


def list_datasets(directory: Path) -> list[Path]:
    datasets = []
    for d in sorted(directory.iterdir()):
        if d.is_dir():
            datasets.append(d)
    return datasets


def load_h5(file: Path) -> dict[str, np.ndarray]:
    with h5py.File(file, "r") as hf:
        return {name: hf[name][()] for name in hf.keys()}


def load_npz(file: Path) -> dict[str, np.ndarray]:
    with np.load(file) as data:
        return {k: data[k] for k in data.files}


def main() -> None:
    st.title("Datasets Dashboard")

    synthetic_dir = DATA_DIR / "synthetic"
    datasets = list_datasets(synthetic_dir)
    if not datasets:
        st.warning("No datasets found")
        return

    dataset_names = [d.name for d in datasets]
    selected_name = st.sidebar.selectbox("Dataset", dataset_names)
    dataset_dir = datasets[dataset_names.index(selected_name)]

    st.header(selected_name)
    st.write(f"Location: `{dataset_dir}`")

    files = list(dataset_dir.glob("*"))
    data_files = [f for f in files if f.suffix in {".h5", ".csv", ".npz"}]

    if not data_files:
        st.info("No data files to display")
        return

    for f in data_files:
        st.subheader(f.name)
        if f.suffix == ".h5":
            data = load_h5(f)
            for name, array in data.items():
                st.markdown(f"**{name}**  shape: {array.shape}  dtype: {array.dtype}")
                if array.ndim == 0:
                    st.write(array.item())
                elif array.ndim == 1:
                    st.line_chart(array)
                elif array.ndim == 2:
                    st.dataframe(pd.DataFrame(array))
                elif array.ndim == 3 and array.shape[-1] in {1, 3}:
                    st.image(array, caption=name, clamp=True)
                else:
                    st.text(array)
        elif f.suffix == ".csv":
            df = pd.read_csv(f)
            st.dataframe(df)
        elif f.suffix == ".npz":
            arrays = load_npz(f)
            for name, array in arrays.items():
                st.markdown(f"**{name}**  shape: {array.shape}  dtype: {array.dtype}")
                if array.ndim <= 2:
                    st.dataframe(pd.DataFrame(array))
                else:
                    st.text(array)


if __name__ == "__main__":
    main()
