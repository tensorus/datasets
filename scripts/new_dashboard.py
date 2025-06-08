import streamlit as st
import h5py
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
from streamlit_extras.add_vertical_space import add_vertical_space # For spacing if needed
from streamlit_extras.stoggle import stoggle # Example import, not used yet but good for extras
# Attempt to import st.divider, fall back to st.markdown if not found (e.g. older streamlit_extras)
try:
    from streamlit_extras.row import row # Example
    from streamlit_extras.markdownlit import markdownlit # Example
    # For st.divider, it's often directly available or via a specific submodule
    # Let's assume st.divider() is a top-level function in recent Streamlit or a common extra
    # If it's not directly available, we will use st.markdown("---")
    # No specific import for st.divider() from streamlit_extras, usually it's st.divider() if available
    # or we use st.markdown("---")
    pass # Keep it simple, will try st.divider() directly
except ImportError:
    print("Some streamlit_extras components not found, will use fallbacks.")


# --- File Path Definitions ---
# These paths point to the HDF5 dataset files used by the dashboard.
# Path(__file__) gives the path to the current script.
# .resolve() makes it an absolute path.
# .parents[1] goes one level up (from 'scripts' to the repository root).
# Then, it navigates into 'data/synthetic/...'.
DEMO_TENSORS_PATH = Path(__file__).resolve().parents[1] / "data" / "synthetic" / "demo_tensors" / "demo_tensors.h5"
NUMERICAL_TENSORS_PATH = Path(__file__).resolve().parents[1] / "data" / "synthetic" / "tensorus_demonstration_datasets" / "numerical" / "random_numerical_tensors.h5"

# --- Data Loading and Helper Functions ---

def decode_h5_attrs(attrs):
    """
    Decodes HDF5 attributes, converting byte strings and specific NumPy array types
    to regular Python strings for better display in Streamlit.
    Handles single byte strings, NumPy arrays of byte strings, and NumPy object arrays
    that might contain byte strings. Also converts string 'None' to Python None.
    """
    decoded_attrs = {}
    for k, v in attrs.items():
        if isinstance(v, bytes):
            decoded_attrs[k] = v.decode('utf-8', errors='replace')
        elif isinstance(v, np.ndarray) and v.dtype.kind == 'S': # Array of byte strings
            decoded_attrs[k] = [s.decode('utf-8', errors='replace') for s in v]
        elif isinstance(v, str) and v == 'None': # Handle 'None' string often from TensorStorage
            decoded_attrs[k] = None
        elif isinstance(v, np.ndarray) and v.dtype.kind == 'O': # Array of objects
            # Attempt to decode if elements are bytes, otherwise convert to list
            if len(v) > 0 and isinstance(v[0], bytes):
                decoded_attrs[k] = [s.decode('utf-8', errors='replace') for s in v]
            else:
                decoded_attrs[k] = v.tolist() # Convert numpy array of objects/strings to Python list
        else:
            decoded_attrs[k] = v # Keep other types as is
    return decoded_attrs

def load_demo_tensors(file_path: Path):
    """
    Loads tensors from the 'demo_tensors.h5' file.
    This file has a flat structure (datasets at the root level).
    """
    if not file_path.exists():
        st.error(f"Dataset file not found: {file_path}")
        return []

    tensors = []
    try:
        with h5py.File(file_path, "r") as hf:
            for name in hf.keys(): # Iterate through top-level datasets
                data = hf[name][()] # Load full tensor data
                attrs = decode_h5_attrs(dict(hf[name].attrs)) # Decode any attributes
                tensors.append({
                    'name': name,
                    'data': data,
                    'shape': data.shape,
                    'dtype': data.dtype,
                    # Provide a default description if metadata is sparse or missing
                    'metadata': attrs if attrs else {'description': 'General tensor from demo_tensors.h5'}
                })
        return tensors
    except Exception as e:
        st.error(f"Error loading {file_path}: {e}")
        return []

def load_numerical_tensors(file_path: Path):
    """
    Loads tensors from the 'random_numerical_tensors.h5' file.
    This file has tensors stored within a 'random_tensors' group.
    It also contains rich metadata as attributes for each tensor.
    """
    if not file_path.exists():
        st.error(f"Dataset file not found: {file_path}")
        return []

    tensors = []
    try:
        with h5py.File(file_path, "r") as hf:
            if "random_tensors" not in hf: # Check for the specific group
                st.error("'random_tensors' group not found in HDF5 file.")
                return []

            group = hf["random_tensors"]
            for tensor_id in group.keys(): # Iterate through tensors in the group (names are UUIDs)
                data = group[tensor_id][()] # Load full tensor data
                attrs = decode_h5_attrs(dict(group[tensor_id].attrs)) # Decode all attributes

                # Use 'description' from metadata as a more user-friendly name if available
                tensor_name = attrs.get('description', tensor_id)
                # If tensor_id was used as name, ensure it's also available in metadata for completeness
                if 'description' not in attrs and 'id' not in attrs :
                     attrs['id'] = tensor_id

                tensors.append({
                    'name': tensor_name,
                    'data': data,
                    'shape': data.shape,
                    'dtype': data.dtype,
                    'metadata': attrs
                })
        return tensors
    except Exception as e:
        st.error(f"Error loading {file_path}: {e}")
        return []

# --- Main Dashboard Application ---
def main():
    st.set_page_config(layout="wide", page_title="Tensorus Dataset Dashboard")
    st.title("Tensorus: Interactive Dataset Dashboard")
    # --- ADDITION: Main Page Introduction ---
    st.markdown("Explore a variety of tensor datasets. Select a dataset file from the sidebar, then choose a specific tensor to view its details and visualizations.")
    # --- END ADDITION ---

    # Sidebar for dataset selection
    st.sidebar.header("Controls")
    # --- ADDITION: Sidebar Guidance ---
    st.sidebar.caption("1. Choose a dataset file. <br>2. Select a tensor from the list that appears in the main panel.", unsafe_allow_html=True)
    # --- END ADDITION ---

    dataset_options = {
        "General Demo Tensors": DEMO_TENSORS_PATH,
        "Specific Numerical Tensors": NUMERICAL_TENSORS_PATH
    }
    selected_dataset_name = st.sidebar.radio(
        "Choose a Dataset File to Explore:",
        options=list(dataset_options.keys())
    )
    selected_file_path = dataset_options[selected_dataset_name]

    # Main content area
    st.header(f"Exploring: {selected_dataset_name}")
    st.caption(f"Data file: `{selected_file_path}`")

    # Load selected dataset
    loaded_tensors = []
    if selected_dataset_name == "General Demo Tensors":
        loaded_tensors = load_demo_tensors(selected_file_path)
    else: # Specific Numerical Tensors
        loaded_tensors = load_numerical_tensors(selected_file_path)

    # Section for selecting a tensor from the loaded dataset
    st.subheader("Available Tensors")
    if not loaded_tensors:
        st.warning(f"No tensors found or loaded from '{selected_dataset_name}'. Review messages above if an error occurred during loading.")
    else:
        tensor_names = [t['name'] for t in loaded_tensors]

        if 'selected_tensor_name_for_dataset' not in st.session_state or \
           st.session_state.get('current_dataset_for_selection') != selected_dataset_name:
            st.session_state.selected_tensor_name_for_dataset = tensor_names[0] if tensor_names else None
            st.session_state.current_dataset_for_selection = selected_dataset_name

        selected_tensor_name = st.selectbox(
            "Select a tensor to view details:",
            options=tensor_names,
            key='selected_tensor_name_for_dataset'
        )

        selected_tensor_data = None
        if selected_tensor_name:
            for t in loaded_tensors:
                if t['name'] == selected_tensor_name:
                    selected_tensor_data = t
                    break

        # --- Tensor Details & Visualization Section ---
        st.subheader("Tensor Details & Visualization")
        if selected_tensor_data:
            with st.container():
                st.markdown(f"### {selected_tensor_data['name']}")

                col1, col2 = st.columns(2)
                with col1:
                    st.markdown(f"**Shape:** `{selected_tensor_data['shape']}`")
                with col2:
                    st.markdown(f"**Data Type:** `{selected_tensor_data['dtype']}`")

                if selected_tensor_data['metadata']:
                    with st.expander("View Detailed Metadata"):
                        st.json(selected_tensor_data['metadata'])
                else:
                    st.markdown("_No additional metadata available._")

            try:
                st.divider()
            except AttributeError:
                st.markdown("---")

            data = selected_tensor_data['data']

            if data.ndim == 0:
                st.metric("Value", data.item())
            elif data.ndim == 1:
                st.line_chart(data)
            elif data.ndim == 2:
                if data.shape[0] * data.shape[1] > 500000 and (data.shape[0] > 1000 or data.shape[1] > 1000):
                     st.write(f"2D Tensor is very large ({data.shape[0]}x{data.shape[1]}). Displaying as text snippet.")
                     st.text(str(data[:5,:5]))
                else:
                    try:
                        fig, ax = plt.subplots()
                        aspect_ratio = 'auto'
                        if data.shape[0] > 0 and data.shape[1] > 0:
                             if data.shape[0] / data.shape[1] > 5 or data.shape[1] / data.shape[0] > 5:
                                 aspect_ratio = 'auto'
                             else:
                                 aspect_ratio = 'equal'

                        cax = ax.imshow(data, cmap='viridis', aspect=aspect_ratio)
                        fig.colorbar(cax)
                        ax.set_title(f"Heatmap of {selected_tensor_data['name']}")
                        st.pyplot(fig)
                        plt.close(fig)
                    except Exception as e:
                        st.error(f"Could not plot 2D tensor: {e}")
                        st.text(str(data[:10,:10]))
            elif data.ndim == 3:
                display_data_3d = data.copy()
                if np.issubdtype(display_data_3d.dtype, np.floating):
                    min_val, max_val = display_data_3d.min(), display_data_3d.max()
                    if max_val > min_val:
                        display_data_3d = (display_data_3d - min_val) / (max_val - min_val)
                    else:
                        display_data_3d = np.zeros_like(display_data_3d)

                if display_data_3d.shape[-1] == 1:
                    st.image(display_data_3d.squeeze(-1) if display_data_3d.ndim == 3 else display_data_3d,
                             caption=f"Image: {selected_tensor_data['name']} (Grayscale)", clamp=True)
                elif display_data_3d.shape[-1] == 3:
                    st.image(display_data_3d, caption=f"Image: {selected_tensor_data['name']} (RGB)", clamp=True)
                else:
                    st.write("3D tensor. Not directly displayable as a standard image (HxWx1 or HxWx3).")
                    st.text(f"Tensor Data Snippet (e.g., data[:2,:2,:2]):\n{str(data[:2,:2,:2])}")

            elif data.ndim >= 4:
                st.write(f"{data.ndim}D Tensor. Direct visualization is complex.")
                snippet_slice = tuple(slice(0, 2) for _ in range(min(data.ndim, 5)))
                st.text(f"Tensor Data Snippet (e.g., data{str(snippet_slice).replace('slice(None, 2, None)', ':2')}):\n{str(data[snippet_slice])}")
            else:
                st.write("Tensor data type or shape not currently handled for visualization.")
                st.text(str(data))

        elif loaded_tensors:
            st.info("Select a tensor from the list above to see its details and visualization here.")
        # If no tensors were loaded at all, the st.warning from "Available Tensors" section suffices.

if __name__ == "__main__":
    main()
