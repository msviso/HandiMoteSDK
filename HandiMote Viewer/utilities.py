from bleak import BleakScanner, BleakClient, BleakError
import struct
from datetime import datetime, timedelta

def parse_motion_data(data):
    if len(data) >= 21:  # Ensure there's enough data for the new format
        qw, qx, qy, qz, timestamp, event = struct.unpack('<4fIB', data[:21])
        return qw, qx, qy, qz, timestamp, event
    return None, None, None, None, None, None  # Default values in case of insufficient data

async def scan_for_devices(filter_name=None):
    devices = await BleakScanner.discover()
    if filter_name:
        filtered_devices = [dev for dev in devices if dev.name and filter_name in dev.name]
    else:
        filtered_devices = devices
    return filtered_devices

async def connect_to_device(device):
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
        self.motion_positions = []  # Add this to store motion positions with timestamps
        self.gray_image_data_records = []
        self.update_image_callback = update_image_callback
        self.update_motion_callback = update_motion_callback

    def handle_gray_level_image_data(self, sender, data):
        if self.gray_image_data_part is None:
            self.gray_image_data_part = data
        else:
            full_data = self.gray_image_data_part + data
            self.gray_image_data_records.append(full_data)
            self.gray_image_data_part = None
            if self.update_image_callback:
                self.update_image_callback(full_data)

    def handle_motion_data(self, sender, data):
        self.motion_data_records.append(data)
        qw, qx, qy, qz, timestamp, event = parse_motion_data(data)
        if qw is not None:
            current_time = datetime.now()
            self.motion_positions.append((qx, qy, qz, current_time))  # Store the position with timestamp
            self.clean_old_positions()  # Clean old positions
        if self.update_motion_callback:
            self.update_motion_callback(data)

    def clean_old_positions(self):
        current_time = datetime.now()
        self.motion_positions = [(x, y, z, t) for (x, y, z, t) in self.motion_positions
                                 if current_time - t < timedelta(seconds=1)]
        # Removing debug output
        # print(f"Current positions after cleaning: {self.motion_positions}")  # Debugging output


    def get_latest_gray_image_data(self):
        return self.gray_image_data_records[-1] if self.gray_image_data_records else None

    def get_latest_motion_data(self):
        return self.motion_data_records[-1] if self.motion_data_records else None

    def get_all_gray_image_data(self):
        return self.gray_image_data_records

    def get_all_motion_data(self):
        return self.motion_data_records
