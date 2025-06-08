readme_content = """
# Real-World Datasets for Tensorus

This directory contains datasets sourced or simulated from real-world scenarios, intended for demonstration and testing purposes with the Tensorus Agentic Tensor database.

Each subdirectory within `real_world/` focuses on a specific type of data that can be converted into tensors. Detailed information about each dataset, including its source, structure, and any preprocessing notes, can be found in the `README.md` file within its respective subdirectory.

## Available Dataset Categories:

### 1. Time Series Data
- **Directory:** [`time_series/`](./time_series/)
- **Description:** Contains datasets that represent sequences of data points indexed in time order.
- **Example:** Global monthly temperature anomalies.
- **Details:** See [`time_series/README.md`](./time_series/README.md)

### 2. Markdown Documents
- **Directory:** [`markdown_documents/`](./markdown_documents/)
- **Description:** Contains a collection of Markdown (.md) files. This data can be used to demonstrate text processing, NLP tasks, or document embedding.
- **Examples:** README files from popular open-source projects.
- **Details:** See [`markdown_documents/README.md`](./markdown_documents/README.md)

### 3. IoT (Internet of Things) Data
- **Directory:** [`iot_data/`](./iot_data/)
- **Description:** Contains simulated data logs from IoT devices. This typically involves time-stamped sensor readings or device statuses.
- **Example:** Simulated smart thermostat logs.
- **Details:** See [`iot_data/README.md`](./iot_data/README.md)

## Usage

These datasets are provided to showcase how different types of real-world information can be structured and prepared for input into Tensorus. They can be used for developing and testing data loading, tensor conversion, and querying functionalities.
"""

output_filepath = "data/real_world/README.md"

with open(output_filepath, 'w') as f:
    f.write(readme_content)

print(f"Successfully created {output_filepath}")
