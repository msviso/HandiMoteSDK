# utilities.py
from bleak import BleakScanner, BleakClient, BleakError

async def scan_for_devices(filter_name=None):
    """Scans for BLE devices and optionally filters them by name."""
    devices = await BleakScanner.discover()
    if filter_name:
        filtered_devices = [dev for dev in devices if dev.name and filter_name in dev.name]
    else:
        filtered_devices = devices
    return filtered_devices

async def connect_to_device(device):
    """Attempt to connect to a single BLE device, where `device` must have a string attribute `address`."""
    if not isinstance(device.address, str):
        print("Device address is not a string:", device.address)
        return None

    print(f"Connecting to device with address: {device.address}")
    client = BleakClient(device.address)
    try:
        await client.connect()
        print(f"Connected to {device.name} at {device.address}.")
        return client
    except Exception as e:
        print(f"Failed to connect to {device.name} at {device.address}: {e}")
        return None


class DataCollector:
    def __init__(self, update_image_callback=None, update_motion_callback=None):
        self.gray_image_data_part = None
        self.motion_data_records = []
        self.gray_image_data_records = []
        self.update_image_callback = update_image_callback  # Callback for image updates
        self.update_motion_callback = update_motion_callback  # Callback for motion updates

    def handle_gray_level_image_data(self, sender, data):
        if self.gray_image_data_part is None:
            self.gray_image_data_part = data
        else:
            full_data = self.gray_image_data_part + data
            self.gray_image_data_records.append(full_data)
            self.gray_image_data_part = None  # Reset for the next frame
            if self.update_image_callback:
                self.update_image_callback(full_data)  # Trigger the image update callback with the new data

    def handle_motion_data(self, sender, data):
        self.motion_data_records.append(data)
        # Trigger the motion update callback with new data
        if self.update_motion_callback:
            self.update_motion_callback(data)  # Convert data appropriately if necessary
        #print(f"Motion data received from {sender}: {data}")

    def get_latest_gray_image_data(self):
        return self.gray_image_data_records[-1] if self.gray_image_data_records else None

    def get_latest_motion_data(self):
        return self.motion_data_records[-1] if self.motion_data_records else None

    def get_all_gray_image_data(self):
        return self.gray_image_data_records

    def get_all_motion_data(self):
        return self.motion_data_records
