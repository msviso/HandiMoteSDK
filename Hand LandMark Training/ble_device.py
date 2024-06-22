from bleak import BleakClient

class BLEDevice:
    """
    Class to manage Bluetooth Low Energy (BLE) device interactions using Bleak.
    """

    def __init__(self, client):
        """
        Initializes the BLEDevice with a BleakClient instance.
        
        Parameters:
        - client: An instance of BleakClient for BLE communication.
        """
        self.client = client

    async def connect(self):
        """
        Connects to the BLE device if not already connected.
        
        Returns:
        - True if connection is successful or already connected, False otherwise.
        """
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
        """
        Disconnects from the BLE device if currently connected.
        """
        if self.client and self.client.is_connected:
            try:
                await self.client.disconnect()
                print("Disconnected successfully.")
            except Exception as e:
                print(f"Failed to disconnect: {e}")
            finally:
                self.client = None
        else:
            print("No active connection to disconnect.")

    async def read_characteristic(self, uuid):
        """
        Reads the value of a BLE characteristic.
        
        Parameters:
        - uuid: The UUID of the characteristic to read.
        
        Returns:
        - The value of the characteristic or None if an error occurs.
        """
        if self.client and self.client.is_connected:
            try:
                return await self.client.read_gatt_char(uuid)
            except Exception as e:
                print(f"Error reading characteristic {uuid}: {e}")
                return None

    async def write_characteristic(self, uuid, data):
        """
        Writes data to a BLE characteristic.
        
        Parameters:
        - uuid: The UUID of the characteristic to write to.
        - data: The data to write to the characteristic.
        """
        if self.client and self.client.is_connected:
            try:
                await self.client.write_gatt_char(uuid, data)
                print("Data written successfully.")
            except Exception as e:
                print(f"Error writing characteristic {uuid}: {e}")

    async def start_notifications(self, uuid, handler):
        """
        Subscribes to notifications for a BLE characteristic.
        
        Parameters:
        - uuid: The UUID of the characteristic to subscribe to.
        - handler: The callback function to handle incoming notifications.
        """
        if self.client and self.client.is_connected:
            try:
                await self.client.start_notify(uuid, handler)
                print(f"Subscribed to notifications for {uuid}.")
            except Exception as e:
                print(f"Error subscribing to notifications: {e}")

    async def stop_notifications(self, uuid):
        """
        Unsubscribes from notifications for a BLE characteristic.
        
        Parameters:
        - uuid: The UUID of the characteristic to unsubscribe from.
        """
        if self.client and self.client.is_connected:
            try:
                await self.client.stop_notify(uuid)
                print(f"Unsubscribed from notifications for {uuid}.")
            except Exception as e:
                print(f"Error unsubscribing from notifications: {e}")

    def is_connected(self):
        """
        Checks if the client is connected to the BLE device.
        
        Returns:
        - True if connected, False otherwise.
        """
        return self.client.is_connected if self.client else False
