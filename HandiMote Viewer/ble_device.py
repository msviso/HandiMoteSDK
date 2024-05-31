# ble_device.py
from bleak import BleakClient

class BLEDevice:
    def __init__(self, client):
        self.client = client

    async def connect(self):
        if not self.client.is_connected:
            try:
                await self.client.connect()
                print("Connected successfully.")
                return True
            except Exception as e:
                print(f"Failed to connect: {e}")
                return False
        else:
            print("Already connected.")
            return True

    async def disconnect(self):
        if self.client and self.client.is_connected:
            try:
                await self.client.disconnect()
                print("Disconnected successfully")
            except Exception as e:
                print(f"Failed to disconnect: {e}")
            finally:
                self.client = None
        else:
            print("No active connection to disconnect.")

    async def read_characteristic(self, uuid):
        if self.client and self.client.is_connected:
            try:
                return await self.client.read_gatt_char(uuid)
            except Exception as e:
                print(f"Error reading characteristic {uuid}: {e}")
                return None

    async def write_characteristic(self, uuid, data):
        if self.client and self.client.is_connected:
            try:
                await self.client.write_gatt_char(uuid, data)
                print("Data written successfully")
            except Exception as e:
                print(f"Error writing characteristic {uuid}: {e}")

    async def start_notifications(self, uuid, handler):
        if self.client and self.client.is_connected:
            try:
                await self.client.start_notify(uuid, handler)
                print(f"Subscribed to notifications for {uuid}")
            except Exception as e:
                print(f"Error subscribing to notifications: {e}")

    async def stop_notifications(self, uuid):
        if self.client and self.client.is_connected:
            try:
                await self.client.stop_notify(uuid)
                print(f"Unsubscribed from notifications for {uuid}")
            except Exception as e:
                print(f"Error unsubscribing from notifications: {e}")

def is_connected(self):
    return self.client.is_connected if self.client else False

