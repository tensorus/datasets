import csv
import datetime
import random
import os

# Define parameters
output_path = "data/real_world/iot_data/"
filename = "thermostat_log.csv"
readme_filename = "README.md"
device_ids = ["thermostat_living_room", "thermostat_bedroom_1", "thermostat_kitchen"]
start_time = datetime.datetime(2023, 10, 1, 0, 0, 0, tzinfo=datetime.timezone.utc)
time_increment_minutes = 15  # Time between readings
num_days = 3 # Simulate data for 3 days
records_per_day_per_device = (24 * 60) // time_increment_minutes
total_records_to_generate_actually = len(device_ids) * records_per_day_per_device * num_days


# --- Create the CSV data file ---
header = ["timestamp", "device_id", "actual_temp_celsius", "target_temp_celsius", "humidity_percent", "status"]
rows = []

current_time = start_time
# Calculate iterations based on num_days and time_increment_minutes for all devices
total_time_steps = records_per_day_per_device * num_days

for i in range(total_time_steps): # Iterate through time steps
    for device_id in device_ids:
        actual_temp = round(random.uniform(18.0, 25.0), 1)
        target_temp = round(random.uniform(20.0, 23.0), 1)
        humidity = round(random.uniform(30.0, 60.0), 1)

        status = "OFF"
        if actual_temp < target_temp - 0.5:
            status = "HEATING"
        elif actual_temp > target_temp + 0.5:
            status = "COOLING"
        elif abs(actual_temp - target_temp) <= 1.0 and random.random() < 0.2: # Occasionally run fan
            status = "FAN_ONLY"

        rows.append([
            current_time.isoformat(),
            device_id,
            actual_temp,
            target_temp,
            humidity,
            status
        ])
    current_time += datetime.timedelta(minutes=time_increment_minutes)

# Ensure correct number of rows by truncating if necessary, or handling the logic inside loop more carefully.
# The script generates one timestamp for all devices, then increments time.
# So, the number of rows will be total_time_steps * len(device_ids)
# The original script had a slight bug in total_records vs how it was used in loop.
# Recalculating actual number of rows based on loop structure
actual_rows_generated = total_time_steps * len(device_ids)


# Ensure the output directory exists (it should from a previous step)
os.makedirs(output_path, exist_ok=True)

# Write the CSV file
csv_file_path = os.path.join(output_path, filename)
with open(csv_file_path, 'w', newline='') as csvfile:
    writer = csv.writer(csvfile)
    writer.writerow(header)
    writer.writerows(rows)

# --- Create the README.md file ---
readme_content = f"""
# IoT Data: Simulated Smart Thermostat Logs

This dataset contains simulated time series data from smart thermostats.

- **File:** `{filename}`
- **Format:** CSV
- **Columns:**
    - `timestamp`: ISO 8601 timestamp of the reading.
    - `device_id`: Unique identifier for the thermostat device.
    - `actual_temp_celsius`: The actual temperature measured by the thermostat, in Celsius.
    - `target_temp_celsius`: The target temperature set for the thermostat, in Celsius.
    - `humidity_percent`: The ambient humidity measured by the thermostat, as a percentage.
    - `status`: The operational status of the HVAC system controlled by the thermostat (e.g., HEATING, COOLING, FAN_ONLY, OFF).

## Data Generation

The data is synthetically generated to simulate typical thermostat readings over a period. It includes multiple devices and various states.
The data covers {num_days} days with readings every {time_increment_minutes} minutes for {len(device_ids)} devices, totaling {actual_rows_generated} records.

## Schema Example:

```
{",".join(header)}
{rows[0][0]},{rows[0][1]},{rows[0][2]},{rows[0][3]},{rows[0][4]},{rows[0][5]}
...
```
"""

readme_file_path = os.path.join(output_path, readme_filename)
with open(readme_file_path, 'w') as rf:
    rf.write(readme_content)

print(f"Successfully created {csv_file_path} with {actual_rows_generated} records and {readme_file_path} in {output_path}")
