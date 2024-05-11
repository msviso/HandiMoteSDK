# HandiMoteSDK without dashboard
Microsense Vision HandiMote SDK example

```
/HandiMoteSDK
|-- ble_device.py
|-- config.py
|-- notification_handler.py
|-- utilities.py
|-- example_usage.py
|-- README.md
|-- setup.py

```

# SDK Documentation

This SDK allows you to interact with a BLE device to collect data from both gray level image matrices and motion sensors. The following sections detail how to start notifications, collect data, and stop notifications.

## Getting Started

Ensure your environment is set up to use the SDK by installing all necessary dependencies:


`pip install bleak`


## Usage
Import SDK Components

First, ensure you import the necessary components from the SDK:

```
from utilities import scan_for_devices, connect_to_device
from ble_device import BLEDevice
from config import UUID_GRAY_LEVEL_IMAGE, UUID_6DOF_MOTION_SENSOR
```

## Initialize Device Connection
Scan for available devices and connect to the desired BLE device:

```
devices = await scan_for_devices("HandiMote")
client = None
if devices:
    client = await connect_to_device(devices[0])  # Connect to the first device found
```

## Start Notifications
Use the BLEDevice class to handle device interactions:

```
if client:
    ble_device = BLEDevice(client)
    if await ble_device.connect():
        print("Device connected successfully.")
```

## Collect Gray Level Image Data
Start receiving data from the gray level image matrix:

```
await ble_device.start_notifications(UUID_GRAY_LEVEL_IMAGE, data_collector.handle_gray_level_image_data)
print("Started notifications for Gray Level Image Data.")
```
## Collect Motion Data
Similarly, start receiving data from the motion sensor:

```
gray_image_data = data_collector.get_all_gray_image_data()
motion_data = data_collector.get_all_motion_data()
```

## Stop Notifications
To stop receiving data from the device:

```
await ble_device.stop_notifications(UUID_GRAY_LEVEL_IMAGE)
await ble_device.stop_notifications(UUID_6DOF_MOTION_SENSOR)
print("Stopped all notifications.")
```

## Disconnect
Finally, ensure you disconnect from the device properly:

```
await ble_device.disconnect()
print("Device disconnected.")
```
## Conclusion
This SDK provides a streamlined approach to interacting with specific BLE devices, facilitating easy data collection from both gray level image sensors and motion sensors.


