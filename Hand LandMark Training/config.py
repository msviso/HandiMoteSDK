# config.py
import re

# UUIDs for the HandiMote device
DEVICE_NAME = "HandiMote"
UUID_GRAY_LEVEL_IMAGE = "f3645566-00b0-4240-ba50-05ca45bf8abc"
UUID_6DOF_MOTION_SENSOR = "f3645571-00b0-4240-ba50-05ca45bf8abc"
UUID_UART_SERVICE = "6e400001-b5a3-f393-e0a9-e50e24dcca9e"
UUID_UART_RX = "6e400002-b5a3-f393-e0a9-e50e24dcca9e"  # Device to receive data
UUID_UART_TX = "6e400003-b5a3-f393-e0a9-e50e24dcca9e"  # Device to send data

def get_device_address(file_path="add.h"):
    with open(file_path, "r") as file:
        content = file.read()
        match = re.search(r'#define\s+DEVICE_ADDRESS\s+"([^"]+)"', content)
        if match:
            return match.group(1)
    return None

DEVICE_ADDRESS = get_device_address()