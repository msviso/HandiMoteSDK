import asyncio
import tkinter as tk
from PIL import Image, ImageTk  # Ensure Pillow is installed with pip install pillow
from utilities import scan_for_devices, connect_to_device, DataCollector
from ble_device import BLEDevice
from config import UUID_GRAY_LEVEL_IMAGE, UUID_6DOF_MOTION_SENSOR
import threading
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import numpy as np
import struct
import time
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import cv2
from PIL import Image, ImageTk  # Ensure Pillow is installed with pip install pillow


# Global definition of image size and scaling factor
SCALING_FACTOR = 20
IMAGE_WIDTH = 24
IMAGE_HEIGHT = 20
NEW_SIZE = (IMAGE_WIDTH * SCALING_FACTOR, IMAGE_HEIGHT * SCALING_FACTOR)

client = None
ble_device = None
# Global variable to hold the cube's current configuration if needed
cube_poly = None

# Main window setup
root = tk.Tk()
root.title("Microsense Vision Dashboard")
root.geometry(f'{NEW_SIZE[0] * 2 + 100}x{NEW_SIZE[1] + 200}')  # Adjust the window size based on the content

# Top label for company name
top_label = tk.Label(root, text="Microsense Vision", font=("Arial", 24))
top_label.pack(side=tk.TOP, pady=10)

root.geometry('800x600')  # Adjust the size as needed based on your scaling

# Middle frame for gray level image and motion cube
middle_frame = tk.Frame(root)
middle_frame.pack(fill=tk.BOTH, expand=True)

# Frame for gray level image
gray_image_frame = tk.Frame(middle_frame, borderwidth=2, relief=tk.SUNKEN)
gray_image_frame.pack(side=tk.LEFT, expand=True, fill=tk.BOTH)
gray_image_frame.config(width=NEW_SIZE[0], height=NEW_SIZE[1])
gray_image_frame.pack_propagate(False)  # Prevents the frame from shrinking to the label size

# Label as a placeholder for gray level image
gray_image_label = tk.Label(gray_image_frame)
gray_image_label.pack(expand=True, fill=tk.BOTH, padx=10, pady=10)

# Frame for motion cube
motion_cube_frame = tk.Frame(middle_frame, borderwidth=2, relief=tk.SUNKEN)
motion_cube_frame.pack(side=tk.LEFT, expand=True, fill=tk.BOTH)

# Label as a placeholder for motion cube
motion_cube_label = tk.Label(motion_cube_frame, text="Motion Cube Display", bg='lightblue')
motion_cube_label.pack(expand=True, fill=tk.BOTH, padx=10, pady=10)

# Bottom frame for messages and buttons
bottom_frame = tk.Frame(root)
bottom_frame.pack(side=tk.BOTTOM, fill=tk.X)

# Message box for information display
message_box = tk.Text(bottom_frame, height=5)
message_box.pack(side=tk.LEFT, padx=10, pady=10, expand=True, fill=tk.X)

# Initialize the matplotlib figure within the motion_cube_frame
fig = plt.figure(figsize=(5, 4))
ax = fig.add_subplot(111, projection='3d')
canvas = FigureCanvasTkAgg(fig, master=motion_cube_frame)
canvas_widget = canvas.get_tk_widget()
canvas_widget.pack(fill=tk.BOTH, expand=True)


class UpdateLimiter:
    def __init__(self, update_interval):
        self.update_interval = update_interval  # Interval in seconds
        self.last_update_time = time.time()

    def should_update(self):
        current_time = time.time()
        if current_time - self.last_update_time >= self.update_interval:
            self.last_update_time = current_time
            return True
        return False
    

# Create an instance of the limiter with desired update interval, e.g., 0.1 seconds
motion_update_limiter = UpdateLimiter(0.1)

def parse_motion_data(data):
    if len(data) >= 12:  # Ensure there's enough data for three floats
        # Unpack three floats from the data; adjust '<3f' depending on actual data structure
        x, y, z = struct.unpack('<3f', data[:12])
        return (x, y, z)  # Return a tuple of the three values
    return (0.0, 0.0, 0.0)  # Default value in case of insufficient data

# Define cube in 3D space
def init_motion_cube(ax):
    r = [-1, 1]
    X, Y, Z = np.meshgrid(r, r, r)
    vertices = np.column_stack([X.flatten(), Y.flatten(), Z.flatten()])
    faces = [[vertices[idx] for idx in face] for face in [
        [0, 1, 3, 2], [4, 5, 7, 6], [0, 1, 5, 4], 
        [2, 3, 7, 6], [0, 2, 6, 4], [1, 3, 7, 5]]]
    cube = Poly3DCollection(faces, facecolors='cyan', linewidths=1, edgecolors='r', alpha=.25)
    ax.add_collection3d(cube)
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_xlim([-2, 2])
    ax.set_ylim([-2, 2])
    ax.set_zlim([-2, 2])

init_motion_cube(ax)  # Initialize cube

def update_motion_cube(data):
    """ Update the position based on data but don't clear the entire axes. """
    x, y, z = parse_motion_data(data)
    if motion_update_limiter.should_update():
        # Add only the new point without clearing the cube
        ax.scatter([x], [y], [z], color='red')
        canvas.draw()

def update_gray_image(image_data):
    # Assuming image_data is raw byte data that needs to be converted into an image
    # Convert byte data to a NumPy array that represents the image
    np_image = np.frombuffer(image_data, dtype=np.uint8).reshape(IMAGE_HEIGHT, IMAGE_WIDTH)
    # Resize the image using OpenCV
    resized_image = cv2.resize(np_image, (NEW_SIZE[0], NEW_SIZE[1]), interpolation=cv2.INTER_NEAREST)
    # Convert the OpenCV image to a PIL image
    pil_image = Image.fromarray(resized_image)
    # Convert the PIL image to a PhotoImage
    photo = ImageTk.PhotoImage(image=pil_image)
    # Update the label
    gray_image_label.config(image=photo)
    gray_image_label.image = photo  # Keep a reference to avoid garbage collection

async def main(loop):
    devices = await scan_for_devices("HandiMote")
    if devices:
        client = await connect_to_device(devices[0])  # Connect to the first device found
        if client:
            ble_device = BLEDevice(client)
            if await ble_device.connect():
                # Start notifications for gray level image data
                await ble_device.start_notifications(UUID_GRAY_LEVEL_IMAGE, data_collector.handle_gray_level_image_data)
                # Start notifications for motion data
                await ble_device.start_notifications(UUID_6DOF_MOTION_SENSOR, data_collector.handle_motion_data)
                print("Receiving gray level image and motion sensor data. Press Ctrl+C to stop...")
                while True:
                    await asyncio.sleep(0.1)  # Sleep to allow other operations
            else:
                print("Failed to establish a proper connection.")
            await ble_device.disconnect()
    else:
        print("No devices found matching the criteria.")


def close_app():
    global client, ble_device
    if client and ble_device:
        asyncio.run_coroutine_threadsafe(shutdown(loop, client, ble_device), loop)
    root.destroy()


async def shutdown(loop, client, ble_device):
    print("Shutting down...")
    try:
        # Stop notifications to ensure no callbacks try to run on a closed loop
        if client and ble_device.is_connected:
            await ble_device.stop_notifications(UUID_GRAY_LEVEL_IMAGE)
        # Cancel all running tasks
        for task in asyncio.all_tasks(loop):
            task.cancel()
        # Stop the loop
        loop.stop()
        while True:
            # Run remaining tasks to completion
            try:
                loop.run_until_complete(loop.shutdown_asyncgens())
                loop.run_until_complete(loop.shutdown_default_executor())
            except asyncio.CancelledError:
                continue  # Ignore CancelledError as we are shutting down
            break
    finally:
        # Finally, close the loop to clean up
        loop.close()
    print("Loop closed.")


def update_message_box(text):
    """ Append text to the message box in a thread-safe manner. """
    def append_text():
        message_box.insert(tk.END, text + "\n")  # Append new text
        message_box.see(tk.END)  # Scroll to the bottom

    root.after(0, append_text)  # Schedule the append_text function to be run by the Tkinter main loop

def update_motion_cube(data):
    """ Update the position based on data and also update the message box. """
    x, y, z = parse_motion_data(data)
    if motion_update_limiter.should_update():
        # Formulate the text to display
        display_text = f"Motion Data - X: {x:.2f}, Y: {y:.2f}, Z: {z:.2f}"
        update_message_box(display_text)  # Update the message box with motion data

        # Update the cube visualization
        ax.scatter([x], [y], [z], color='red')  # This adds a new red point for each update
        canvas.draw()

init_motion_cube(ax)  # Initialize the 3D cube in the plot

# Button area with a close window button
close_button = tk.Button(bottom_frame, text="Close Window", command=close_app)
close_button.pack(side=tk.RIGHT, padx=10, pady=10)

data_collector = DataCollector(update_image_callback=update_gray_image, update_motion_callback=update_motion_cube)

# Start the asyncio event loop in a separate thread
def start_asyncio_loop():
    asyncio.set_event_loop(loop)
    try:
        loop.run_forever()
    finally:
        loop.close()

loop = asyncio.new_event_loop()
thread = threading.Thread(target=start_asyncio_loop)
thread.start()

# Properly schedule the main function to run
asyncio.run_coroutine_threadsafe(main(loop), loop)
root.protocol("WM_DELETE_WINDOW", close_app)  # Properly handle window closing
root.mainloop()