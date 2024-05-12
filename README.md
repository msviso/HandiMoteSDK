
# SDK Documentation

This SDK allows you to interact with a BLE device to collect data from both gray level image matrices and motion sensors. The following sections detail how to start notifications, collect data, and stop notifications.

## Getting Started

Ensure your environment is set up to use the SDK by installing all necessary dependencies:


`pip install bleak`



## SDK Examples
  ### HandiMoteSDK without Dashboard
  Microsense Vision HandiMote SDK example
  
  Description : 
  Basic handimote python data connect example SDK.
  ```
  /HandiMoteSDK/HandiMoteConnect_Without_Dashboard/
  |-- ble_device.py
  |-- config.py
  |-- notification_handler.py
  |-- utilities.py
  |-- example_usage.py
  |-- README.md
  |-- setup.py
  
  ```

## Setting the BLE Address
The BLE address is unique to your Bluetooth device. You need to replace the placeholder with the actual address of your HandiMote device in the config.py file.

1. Locate the Device Address:

Use a Bluetooth scanner on your phone or computer to find the BLE address of your HandiMote. It usually looks something like XX:XX:XX:XX:XX:XX.

2. Modify the Config File:

* Open the config.py file located in the root of the project directory.
* Find the line that defines the ADDRESS variable. It will look something like this:
python

``` python
ADDRESS = "Device_MAC_Address"  # Default, replace with your device's MAC address
```
* Replace "Device_MAC_Address" with the BLE address of your device
``` python
ADDRESS = "XX:XX:XX:XX:XX:XX"  # Example: ADDRESS = "01:23:45:67:89:AB"
```
