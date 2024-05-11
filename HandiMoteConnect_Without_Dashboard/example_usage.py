import asyncio
from utilities import scan_for_devices, connect_to_device, DataCollector
from ble_device import BLEDevice
from config import UUID_GRAY_LEVEL_IMAGE, UUID_6DOF_MOTION_SENSOR

async def main():
    devices = await scan_for_devices("HandiMote")
    if devices:
        client = await connect_to_device(devices[0])  # Connect to the first device found
        if client:
            ble_device = BLEDevice(client)
            data_collector = DataCollector()
            if await ble_device.connect():
                await ble_device.start_notifications(UUID_GRAY_LEVEL_IMAGE, data_collector.handle_gray_level_image_data)
                await ble_device.start_notifications(UUID_6DOF_MOTION_SENSOR, data_collector.handle_motion_data)
                print("Receiving gray level image and motion sensor data. Press Ctrl+C to stop...")
                try:
                    # Run for a specific duration or until manual stop
                    for _ in range(30):  # Example: run for 30 seconds
                        if data_collector.get_latest_gray_image_data():
                            print("Latest Gray Image Data:", data_collector.get_latest_gray_image_data())
                        if data_collector.get_latest_motion_data():
                            print("Latest Motion Data:", data_collector.get_latest_motion_data())
                        await asyncio.sleep(1)
                except KeyboardInterrupt:
                    print("Stopping notifications and disconnecting...")
                finally:
                    await ble_device.stop_notifications(UUID_GRAY_LEVEL_IMAGE)
                    await ble_device.stop_notifications(UUID_6DOF_MOTION_SENSOR)
            else:
                print("Failed to establish a proper connection.")
            await ble_device.disconnect()
    else:
        print("No devices found matching the criteria.")

if __name__ == "__main__":
    asyncio.run(main())
