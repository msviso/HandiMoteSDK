from bleak import BleakScanner, BleakClient, BleakError
from motion_processor import parse_motion_data
import time

# Function to scan for BLE devices
async def scan_for_devices(filter_name=None):
    # Discover all nearby BLE devices
    devices = await BleakScanner.discover()
    if filter_name:
        # Filter devices by name if a filter is provided
        filtered_devices = [dev for dev in devices if dev.name and filter_name in dev.name]
    else:
        # Otherwise, return all discovered devices
        filtered_devices = devices
    return filtered_devices

# Function to connect to a specific BLE device
async def connect_to_device(device):
    # Ensure the device address is a string
    if not isinstance(device.address, str):
        print("Device address is not a string:", device.address)
        return None

    print(f"Connecting to device with address: {device.address}")
    client = BleakClient(device.address)
    try:
        # Attempt to connect to the device
        await client.connect()
        print(f"Connected to {device.name} at {device.address}.")
        return client
    except Exception as e:
        # Handle connection failure
        print(f"Failed to connect to {device.name} at {device.address}: {e}")
        return None

# Class to collect and process data from the BLE device
class DataCollector:
    def __init__(self, update_image_callback=None, update_motion_callback=None):
        self.gray_image_data_part = None  # Part of the gray image data
        self.motion_positions = []  # Store quaternion and timestamp
        self.gray_image_data_records = []  # Store full gray image data records
        self.update_image_callback = update_image_callback  # Callback for updating images
        self.update_motion_callback = update_motion_callback  # Callback for updating motion data

    # Handle receiving gray level image data
    def handle_gray_level_image_data(self, sender, data):
        if self.gray_image_data_part is None:
            # If there is no existing data part, start a new one
            self.gray_image_data_part = data
        else:
            # If there is existing data part, combine it with new data
            full_data = self.gray_image_data_part + data
            self.gray_image_data_records.append(full_data)  # Save the full data
            self.gray_image_data_part = None  # Reset the data part
            if self.update_image_callback:
                self.update_image_callback(full_data)  # Call the image update callback

    # Handle receiving motion data
    def handle_motion_data(self, sender, data):
        parsed_data = parse_motion_data(data)  # Parse the motion data
        if parsed_data:
            # Unpack the parsed data
            qw, qx, qy, qz, timestamp, event = parsed_data
            self.motion_positions.append((qw, qx, qy, qz, timestamp))
            if len(self.motion_positions) > 10:  # Process every 10 data points
                self.clean_old_positions()  # Clean old positions
                if self.update_motion_callback:
                    self.update_motion_callback(data)  # Call the motion update callback

    # Clean old motion positions
    def clean_old_positions(self):
        current_time = time.time()
        # Keep only the positions from the last second
        self.motion_positions = [(qw, qx, qy, qz, t) for (qw, qx, qy, qz, t) in self.motion_positions
                                if current_time - t < 1]

    # Get the latest gray image data
    def get_latest_gray_image_data(self):
        return self.gray_image_data_records[-1] if self.gray_image_data_records else None

    # Get the latest motion data
    def get_latest_motion_data(self):
        return self.motion_positions[-1] if self.motion_positions else None

    # Get all gray image data records
    def get_all_gray_image_data(self):
        return self.gray_image_data_records

    # Get all motion data records
    def get_all_motion_data(self):
        return self.motion_positions
