import asyncio
import numpy as np
import threading
from utilities import scan_for_devices, connect_to_device, DataCollector
from ble_device import BLEDevice
from config import UUID_GRAY_LEVEL_IMAGE, UUID_6DOF_MOTION_SENSOR
from opencv_processor import process_image
from motion_processor import parse_motion_data, quaternion_to_rotation_matrix, quaternion_multiply
from datetime import datetime
import os
import sys
from PyQt5 import QtWidgets, QtGui, QtCore
import time
import cv2
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

# Global definition of image size and scaling factor
SCALING_FACTOR = 20
IMAGE_WIDTH = 24
IMAGE_HEIGHT = 20
NEW_SIZE = (IMAGE_WIDTH * SCALING_FACTOR, IMAGE_HEIGHT * SCALING_FACTOR)

client = None
ble_device = None
cube_poly = None

current_image_data = None
motion_positions = []

class UpdateLimiter:
    """Class to limit the update rate of the motion cube."""
    def __init__(self, update_interval):
        self.update_interval = update_interval
        self.last_update_time = time.time()

    def should_update(self):
        current_time = time.time()
        if current_time - self.last_update_time >= self.update_interval:
            self.last_update_time = current_time
            return True
        return False

motion_update_limiter = UpdateLimiter(0.05)  # Adjusted update interval for smoother path

class MotionCubeCanvas(FigureCanvas):
    def __init__(self, parent=None, width=5, height=4, dpi=100):
        self.fig = Figure(figsize=(width, height), dpi=dpi)
        self.ax = self.fig.add_subplot(111, projection='3d')
        super(MotionCubeCanvas, self).__init__(self.fig)
        self.setParent(parent)

        self.init_cube()

    def init_cube(self):
        r = [-1, 1]
        X, Y, Z = np.meshgrid(r, r, r)
        self.vertices = np.column_stack([X.flatten(), Y.flatten(), Z.flatten()])
        self.faces = [[self.vertices[idx] for idx in face] for face in [
            [0, 1, 3, 2], [4, 5, 7, 6], [0, 1, 5, 4],
            [2, 3, 7, 6], [0, 2, 6, 4], [1, 3, 7, 5]]]
        self.cube = Poly3DCollection(self.faces, facecolors='cyan', linewidths=1, edgecolors='r', alpha=.25)
        self.ax.add_collection3d(self.cube)
        self.ax.set_xlim([-2, 2])
        self.ax.set_ylim([-2, 2])
        self.ax.set_zlim([-2, 2])
        self.ax.set_xticks([])  # Disable x-axis ticks
        self.ax.set_yticks([])  # Disable y-axis ticks
        self.ax.set_zticks([])  # Disable z-axis ticks
        self.ax.set_xlabel('')
        self.ax.set_ylabel('')
        self.ax.set_zlabel('')
        self.ax.view_init(elev=20, azim=30)
        self.motion_path = []
        self.current_rotation_matrix = np.identity(3)  # Identity matrix for default rotation

    def draw_cube(self, rotation_matrix=None):
        if rotation_matrix is None:
            rotation_matrix = self.current_rotation_matrix
        else:
            self.current_rotation_matrix = rotation_matrix

        rotated_vertices = np.dot(self.vertices, rotation_matrix.T)
        faces = [[rotated_vertices[j] for j in face] for face in [
            [0, 1, 3, 2], [4, 5, 7, 6], [0, 1, 5, 4],
            [2, 3, 7, 6], [0, 2, 6, 4], [1, 3, 7, 5]]]
        self.cube.set_verts(faces)
        self.draw_motion_path()
        self.fig.canvas.draw()

    def draw_motion_path(self):
        if self.motion_path:
            x, y, z = zip(*self.motion_path)
            self.ax.plot(x, y, z, color='red')

    def update_motion_path(self, position):
        self.motion_path.append(position)
        if len(self.motion_path) > 100:  # Limit the length of the path for performance reasons
            self.motion_path.pop(0)

    def clear_motion_path(self):
        self.motion_path = []
        self.ax.clear()
        self.init_cube()

class MainWindow(QtWidgets.QMainWindow):
    def __init__(self):
        super().__init__()

        self.setWindowTitle("HandiMote Viewer")
        self.setGeometry(100, 100, 1200, 600)

        central_widget = QtWidgets.QWidget()
        self.setCentralWidget(central_widget)

        layout = QtWidgets.QVBoxLayout(central_widget)

        h_layout = QtWidgets.QHBoxLayout()
        layout.addLayout(h_layout)

        self.gray_image_label = QtWidgets.QLabel(self)
        self.gray_image_label.setFixedSize(NEW_SIZE[0], NEW_SIZE[1])
        h_layout.addWidget(self.gray_image_label)

        self.motion_cube_canvas = MotionCubeCanvas(self, width=5, height=4, dpi=100)
        h_layout.addWidget(self.motion_cube_canvas)

        self.capture_button = QtWidgets.QPushButton("Capture Image", self)
        self.capture_button.clicked.connect(self.capture_image)
        layout.addWidget(self.capture_button)

        self.quit_button = QtWidgets.QPushButton("Quit", self)
        self.quit_button.clicked.connect(self.close_app)
        layout.addWidget(self.quit_button)

        self.message_box = QtWidgets.QTextEdit(self)
        self.message_box.setReadOnly(True)
        layout.addWidget(self.message_box)

        self.timer = QtCore.QTimer()
        self.timer.timeout.connect(self.update_display)
        self.timer.start(100)

        self.clear_timer = QtCore.QTimer()
        self.clear_timer.timeout.connect(self.clear_motion_path)
        self.clear_timer.start(3000)  # Clear path every 2 seconds

    def update_display(self):
        pass

    def update_gray_image(self, image_data):
        """Update the gray level image display."""
        global current_image_data

        np_image = np.frombuffer(image_data, dtype=np.uint8).reshape(IMAGE_HEIGHT, IMAGE_WIDTH)
        resized_image = cv2.resize(np_image, (NEW_SIZE[0], NEW_SIZE[1]), interpolation=cv2.INTER_NEAREST)
        q_image = QtGui.QImage(resized_image.data, NEW_SIZE[0], NEW_SIZE[1], QtGui.QImage.Format_Grayscale8)
        pixmap = QtGui.QPixmap.fromImage(q_image)
        self.gray_image_label.setPixmap(pixmap)

        current_image_data = image_data

    def update_motion_cube(self, data):
        """Update the motion cube with new data."""
        qw, qx, qy, qz, timestamp, event = parse_motion_data(data)
        if qw is not None and motion_update_limiter.should_update():
            q = (qw, qx, qy, qz)
            rotation_matrix = quaternion_to_rotation_matrix(*q)
            self.motion_cube_canvas.draw_cube(rotation_matrix)

            # Assuming position can be derived from the quaternion (for example purposes)
            position = (qx, qy, qz)
            self.motion_cube_canvas.update_motion_path(position)

    def clear_motion_path(self):
        """Clear the motion path."""
        self.motion_cube_canvas.clear_motion_path()

    def capture_image(self):
        """Capture the current gray level image and save it."""
        global current_image_data
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        folder_name = "captured_images"
        bin_filename = os.path.join(folder_name, f"vision_image_{timestamp}.bin")
        png_filename = os.path.join(folder_name, f"vision_image_{timestamp}.png")
        
        if not os.path.exists(folder_name):
            os.makedirs(folder_name)
        
        with open(bin_filename, 'wb') as bin_file:
            bin_file.write(current_image_data)
        
        np_image = np.frombuffer(current_image_data, dtype=np.uint8).reshape(IMAGE_HEIGHT, IMAGE_WIDTH)
        resized_image = cv2.resize(np_image, (NEW_SIZE[0], NEW_SIZE[1]), interpolation=cv2.INTER_NEAREST)
        cv2.imwrite(png_filename, resized_image)

        self.print_message(f"Captured image saved as {bin_filename} and {png_filename}")

    def close_app(self):
        """Close the application cleanly."""
        global client, ble_device
        if client and ble_device:
            asyncio.run_coroutine_threadsafe(shutdown(loop, client, ble_device), loop).result()
        self.close()
        loop.call_soon_threadsafe(loop.stop)

    def print_message(self, message):
        """Print a message to the message box."""
        def append_message():
            self.message_box.append(message)
        QtCore.QMetaObject.invokeMethod(self.message_box, "append", QtCore.Qt.QueuedConnection, QtCore.Q_ARG(str, message))

async def main(loop, window):
    """Main function to scan and connect to BLE device."""
    device_found = False
    while not device_found:
        devices = await scan_for_devices("HandiMote")
        if devices:
            device_found = True
            client = await connect_to_device(devices[0])
            if client:
                ble_device = BLEDevice(client)
                if await ble_device.connect():
                    await ble_device.start_notifications(UUID_GRAY_LEVEL_IMAGE, data_collector.handle_gray_level_image_data)
                    await ble_device.start_notifications(UUID_6DOF_MOTION_SENSOR, data_collector.handle_motion_data)
                    while True:
                        await asyncio.sleep(0.1)
                await ble_device.disconnect()
        else:
            window.print_message("Device not found. Retrying in 5 seconds...")
            await asyncio.sleep(5)  # Wait 5 seconds before retrying

async def shutdown(loop, client, ble_device):
    """Shutdown the asyncio loop and cleanup."""
    print("Shutting down...")
    try:
        if client and ble_device.is_connected:
            await ble_device.stop_notifications(UUID_GRAY_LEVEL_IMAGE)
            await ble_device.stop_notifications(UUID_6DOF_MOTION_SENSOR)
            await ble_device.disconnect()
            await asyncio.sleep(1)
        tasks = [task for task in asyncio.all_tasks(loop) if not task.done()]
        for task in tasks:
            task.cancel()
            try:
                await task
            except asyncio.CancelledError:
                pass
        loop.stop()
        while not loop.is_closed():
            loop.run_until_complete(loop.shutdown_asyncgens())
            loop.run_until_complete(loop.shutdown_default_executor())
    finally:
        loop.close()
    print("Loop closed.")

def start_asyncio_loop():
    """Start the asyncio loop."""
    asyncio.set_event_loop(loop)
    try:
        loop.run_forever()
    finally:
        loop.close()

def main_function():
    """Main function to start the application and asyncio loop."""
    global loop, thread, app, window
    app = QtWidgets.QApplication(sys.argv)
    window = MainWindow()
    window.show()

    loop = asyncio.new_event_loop()
    thread = threading.Thread(target=start_asyncio_loop)
    thread.start()
    asyncio.run_coroutine_threadsafe(main(loop, window), loop)

    sys.exit(app.exec_())

if __name__ == "__main__":
    data_collector = DataCollector(update_image_callback=lambda data: window.update_gray_image(data), 
                                   update_motion_callback=lambda data: window.update_motion_cube(data))
    main_function()
