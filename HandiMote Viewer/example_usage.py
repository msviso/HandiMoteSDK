import asyncio
import tkinter as tk
from PIL import Image, ImageTk  # Ensure Pillow is installed with pip install pillow
from utilities import scan_for_devices, connect_to_device, DataCollector
from ble_device import BLEDevice
from config import DEVICE_ADDRESS, UUID_GRAY_LEVEL_IMAGE, UUID_6DOF_MOTION_SENSOR
import threading
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import numpy as np
import struct
import time
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import cv2
from datetime import datetime
from opencv_processor import process_image, match_templates  # Import the new opencv_processor
from motion_processor import parse_motion_data, quaternion_to_rotation_matrix  # Import the new motion_processor
import os

# Global definition of image size and scaling factor
SCALING_FACTOR = 20
IMAGE_WIDTH = 24
IMAGE_HEIGHT = 20
NEW_SIZE = (IMAGE_WIDTH * SCALING_FACTOR, IMAGE_HEIGHT * SCALING_FACTOR)

client = None
ble_device = None
cube_poly = None

root = tk.Tk()
root.title("Microsense Vision Dashboard")
root.geometry(f'{NEW_SIZE[0] * 2 + 100}x{NEW_SIZE[1] + 400}')

top_label = tk.Label(root, text="Microsense Vision", font=("Arial", 24))
top_label.pack(side=tk.TOP, pady=10)

root.geometry('800x800')

middle_frame = tk.Frame(root)
middle_frame.pack(fill=tk.BOTH, expand=True)

gray_image_frame = tk.Frame(middle_frame, borderwidth=2, relief=tk.SUNKEN)
gray_image_frame.pack(side=tk.LEFT, expand=True, fill=tk.BOTH)
gray_image_frame.config(width=NEW_SIZE[0], height=NEW_SIZE[1])
gray_image_frame.pack_propagate(False)

gray_image_label = tk.Label(gray_image_frame)
gray_image_label.pack(expand=True, fill=tk.BOTH, padx=10, pady=10)

motion_cube_frame = tk.Frame(middle_frame, borderwidth=2, relief=tk.SUNKEN)
motion_cube_frame.pack(side=tk.LEFT, expand=True, fill=tk.BOTH)

motion_cube_label = tk.Label(motion_cube_frame, text="Motion Cube Display", bg='lightblue')
motion_cube_label.pack(expand=True, fill=tk.BOTH, padx=10, pady=10)

bottom_frame = tk.Frame(root)
bottom_frame.pack(side=tk.BOTTOM, fill=tk.X)

# 为每个事件添加单独的消息框
event_frames = {}
event_messages = ["Fingers Vision Events", "6DoF Motion Message"]

for event in event_messages:
    frame = tk.Frame(bottom_frame, borderwidth=2, relief=tk.SUNKEN)
    frame.pack(side=tk.TOP, fill=tk.BOTH, expand=True, padx=10, pady=5)
    label = tk.Label(frame, text=event, font=("Arial", 12), bg='lightgray')
    label.pack(side=tk.TOP, fill=tk.X)
    message_box = tk.Text(frame, height=4)
    message_box.pack(side=tk.BOTTOM, fill=tk.BOTH, expand=True, padx=10, pady=5)
    event_frames[event] = message_box

fig = plt.figure(figsize=(5, 4))
ax = fig.add_subplot(111, projection='3d')
canvas = FigureCanvasTkAgg(fig, master=motion_cube_frame)
canvas_widget = canvas.get_tk_widget()
canvas_widget.pack(fill=tk.BOTH, expand=True)

class UpdateLimiter:
    def __init__(self, update_interval):
        self.update_interval = update_interval
        self.last_update_time = time.time()

    def should_update(self):
        current_time = time.time()
        if current_time - self.last_update_time >= self.update_interval:
            self.last_update_time = current_time
            return True
        return False

motion_update_limiter = UpdateLimiter(0.2)

def init_motion_cube(ax, scale=2):
    r = [-1 * scale, 1 * scale]
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
    ax.set_xlim([-2 * scale, 2 * scale])
    ax.set_ylim([-2 * scale, 2 * scale])
    ax.set_zlim([-2 * scale, 2 * scale])

init_motion_cube(ax)

# 當前圖像數據變量
current_image_data = None

def update_motion_cube(data):
    qw, qx, qy, qz, timestamp, event = parse_motion_data(data)
    if qw is not None and motion_update_limiter.should_update():
        display_text = (f"Quaternion - qw: {qw:.2f}, qx: {qx:.2f}, qy: {qy:.2f}, qz: {qz:.2f}\n"
                        f"Timestamp: {timestamp}\nEvent: {event}")
        update_message_box(display_text)

        rotation_matrix = quaternion_to_rotation_matrix(qw, qx, qy, qz)
        apply_rotation_to_cube(rotation_matrix, scale=2)
        
        draw_motion_path(data_collector.motion_positions, scale=2)
        canvas.draw()

def draw_motion_path(positions, scale=2):
    if len(positions) < 2:
        return  # Need at least two points to draw a path

    valid_positions = []
    for pos in positions:
        if len(pos) == 4:
            x, y, z, t = pos
            valid_positions.append((x * scale, y * scale, z * scale))
        # Removing debug output
        # else:
        #     print(f"Invalid position format: {pos}")

    if not valid_positions:
        return

    xs, ys, zs = zip(*valid_positions)
    ax.plot(xs, ys, zs, color='blue')
    # Removing debug output
    # print(f"Plotted path with positions: {valid_positions}")

def apply_rotation_to_cube(rotation_matrix, scale=2):
    global cube_poly
    ax.cla()

    r = [-1 * scale, 1 * scale]
    vertices = np.array([[x, y, z] for x in r for y in r for z in r])
    rotated_vertices = vertices @ rotation_matrix.T

    faces = [[rotated_vertices[j] for j in [0, 1, 3, 2]],
             [rotated_vertices[j] for j in [4, 5, 7, 6]],
             [rotated_vertices[j] for j in [0, 1, 5, 4]],
             [rotated_vertices[j] for j in [2, 3, 7, 6]],
             [rotated_vertices[j] for j in [0, 2, 6, 4]],
             [rotated_vertices[j] for j in [1, 3, 7, 5]]]

    cube_poly = Poly3DCollection(faces, facecolors='cyan', linewidths=1, edgecolors='r', alpha=.25)
    ax.add_collection3d(cube_poly)
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_xlim([-2 * scale, 2 * scale])
    ax.set_ylim([-2 * scale, 2 * scale])
    ax.set_zlim([-2 * scale, 2 * scale])
    ax.view_init(30, 30)

def capture_image():
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    folder_name = "captured_images"
    bin_filename = os.path.join(folder_name, f"vision_image_{timestamp}.bin")
    png_filename = os.path.join(folder_name, f"vision_image_{timestamp}.png")
    
    # if folder not exists create one
    if not os.path.exists(folder_name):
        os.makedirs(folder_name)
    
    # Saving Bin file
    with open(bin_filename, 'wb') as bin_file:
        bin_file.write(current_image_data)
    
    # 保存 PNG 文件
    np_image = np.frombuffer(current_image_data, dtype=np.uint8).reshape(IMAGE_HEIGHT, IMAGE_WIDTH)
    resized_image = cv2.resize(np_image, (NEW_SIZE[0], NEW_SIZE[1]), interpolation=cv2.INTER_NEAREST)
    pil_image = Image.fromarray(resized_image)
    pil_image.save(png_filename)

    update_message_box(f"Captured image saved as {bin_filename} and {png_filename}")

current_image_data = None

def update_gray_image(image_data):
    global current_image_data
    if not root.winfo_exists():
        return  # Exit if root window does not exist

    # Process the image with OpenCV
    processed_image = process_image(image_data, IMAGE_HEIGHT, IMAGE_WIDTH)
    
    # Perform template matching with multiple templates
    templates = ["open.png", "point.png", "point1.png"]  # image match referece
    match_results = match_templates(image_data, IMAGE_HEIGHT, IMAGE_WIDTH, templates)
    if match_results:
        vision_event_texts = []
        for template_path, max_val, max_loc in match_results:
            event_name = os.path.basename(template_path)
            message = f"Template {event_name} match value: {max_val:.2f}, location: {max_loc}"
            vision_event_texts.append(message)
        

        combined_vision_event_text = "\n".join(vision_event_texts)
        update_event_message_box("Fingers Vision Events", combined_vision_event_text)

    np_image = np.frombuffer(image_data, dtype=np.uint8).reshape(IMAGE_HEIGHT, IMAGE_WIDTH)
    resized_image = cv2.resize(np_image, (NEW_SIZE[0], NEW_SIZE[1]), interpolation=cv2.INTER_NEAREST)
    pil_image = Image.fromarray(resized_image)
    photo = ImageTk.PhotoImage(image=pil_image)
    if gray_image_label.winfo_exists():
        gray_image_label.config(image=photo)
        gray_image_label.image = photo

    # 更新當前圖像數據
    current_image_data = image_data

def update_event_message_box(event_name, text):
    """ 更新指定事件的消息框 """
    if event_name in event_frames:
        event_frames[event_name].delete(1.0, tk.END)  # Clear 
        event_frames[event_name].insert(tk.END, text)  # inset the message

async def main(loop):
    devices = await scan_for_devices("HandiMote")
    if devices:
        client = await connect_to_device(devices[0])
        if client:
            ble_device = BLEDevice(client)
            if await ble_device.connect():
                await ble_device.start_notifications(UUID_GRAY_LEVEL_IMAGE, data_collector.handle_gray_level_image_data)
                await ble_device.start_notifications(UUID_6DOF_MOTION_SENSOR, data_collector.handle_motion_data)
                while True:
                    await asyncio.sleep(0.1)
            await ble_device.disconnect()

def close_app():
    global client, ble_device
    if client and ble_device:
        asyncio.run_coroutine_threadsafe(shutdown(loop, client, ble_device), loop).result()
    root.quit()  # Stop the Tkinter mainloop
    root.destroy()
    loop.call_soon_threadsafe(loop.stop)  # Ensure the asyncio loop stops

async def shutdown(loop, client, ble_device):
    print("Shutting down...")
    try:
        # Stop notifications to ensure no callbacks try to run on a closed loop
        if client and ble_device.is_connected:
            await ble_device.stop_notifications(UUID_GRAY_LEVEL_IMAGE)
            await ble_device.stop_notifications(UUID_6DOF_MOTION_SENSOR)
            await ble_device.disconnect()
            await asyncio.sleep(1)  # Add a delay to ensure BLE notifications are fully stopped
        # Cancel all running tasks
        tasks = [task for task in asyncio.all_tasks(loop) if not task.done()]
        for task in tasks:
            task.cancel()
            try:
                await task
            except asyncio.CancelledError:
                pass
        # Stop the loop
        loop.stop()
        while not loop.is_closed():
            loop.run_until_complete(loop.shutdown_asyncgens())
            loop.run_until_complete(loop.shutdown_default_executor())
    finally:
        # Finally, close the loop to clean up
        loop.close()
    print("Loop closed.")

def update_message_box(text):
    """ Append text to the message box in a thread-safe manner. """
    def append_text():
        if not root.winfo_exists():
            return  # Exit if root window does not exist
        if message_box.winfo_exists():
            message_box.insert(tk.END, text + "\n")  # Append new text
            message_box.see(tk.END)  # Scroll to the bottom

    root.after(0, append_text)  # Schedule the append_text function to be run by the Tkinter main loop

def quaternion_to_euler(qw, qx, qy, qz):
    import math
    sinr_cosp = 2 * (qw * qx + qy * qz)
    cosr_cosp = 1 - 2 * (qx * qx + qy * qy)
    roll = math.atan2(sinr_cosp, cosr_cosp)

    sinp = 2 * (qw * qy - qz * qx)
    pitch = math.asin(sinp) if abs(sinp) <= 1 else math.copysign(math.pi / 2, sinp)

    siny_cosp = 2 * (qw * qz + qx * qy)
    cosy_cosp = 1 - 2 * (qy * qy + qz * qz)
    yaw = math.atan2(siny_cosp, cosy_cosp)

    return roll, pitch, yaw

init_motion_cube(ax)

close_button = tk.Button(bottom_frame, text="Close Window", command=close_app)
close_button.pack(side=tk.RIGHT, padx=10, pady=10)

capture_button = tk.Button(bottom_frame, text="Capture Vision", command=capture_image)
capture_button.pack(side=tk.RIGHT, padx=10, pady=10)

data_collector = DataCollector(update_image_callback=update_gray_image, update_motion_callback=update_motion_cube)

def start_asyncio_loop():
    asyncio.set_event_loop(loop)
    try:
        loop.run_forever()
    finally:
        loop.close()

loop = asyncio.new_event_loop()
thread = threading.Thread(target=start_asyncio_loop)
thread.start()

asyncio.run_coroutine_threadsafe(main(loop), loop)
root.protocol("WM_DELETE_WINDOW", close_app)
root.mainloop()