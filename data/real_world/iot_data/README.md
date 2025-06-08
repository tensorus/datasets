
# IoT Data: Simulated Smart Thermostat Logs

This dataset contains simulated time series data from smart thermostats.

- **File:** `thermostat_log.csv`
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
The data covers 3 days with readings every 15 minutes for 3 devices, totaling 864 records.

## Schema Example:

```
timestamp,device_id,actual_temp_celsius,target_temp_celsius,humidity_percent,status
2023-10-01T00:00:00+00:00,thermostat_living_room,23.9,20.0,54.8,COOLING
...
```
