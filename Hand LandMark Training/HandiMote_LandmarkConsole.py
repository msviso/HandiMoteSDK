import asyncio
import numpy as np
import threading
import cv2
from datetime import datetime
from PyQt5 import QtWidgets, QtGui, QtCore
import os
import sys
from utilities import scan_for_devices, connect_to_device, DataCollector
from ble_device import BLEDevice
from config import UUID_GRAY_LEVEL_IMAGE, UUID_6DOF_MOTION_SENSOR
import tensorflow as tf
import tkinter as tk
from tkinter import filedialog
from pyquaternion import Quaternion
from motion_processor import parse_motion_data, quaternion_to_rotation_matrix

# Global definition of image size and scaling factor
SCALING_FACTOR = 20
IMAGE_WIDTH = 24
IMAGE_HEIGHT = 17  # 修改有效像素高度為 17
NEW_SIZE = (IMAGE_WIDTH * SCALING_FACTOR, IMAGE_HEIGHT * SCALING_FACTOR)
THRESHOLD_A = 100  # 設置閥值A
THRESHOLD_B = 150  # 設置閥值B

client = None
ble_device = None
current_image_data = None
motion_data_list = []
landmark_data = None
reference_distance = 6.0  # 參考距離，設定為6cm

# Define custom partitions
combined_partitions = [
    (0, 6, 0, 17),   # Combine Partition 1 and 6
    (6, 11, 0, 12),  # Partition 2
    (11, 15, 0, 12),  # Partition 3
    (15, 19, 0, 12),  # Partition 4
    (19, 24, 0, 12),  # Partition 5
    (5, 11, 12, 17),  # Partition 7
    (11, 16, 12, 17),  # Partition 8
    (16, 20, 12, 17),  # Partition 9
    (20, 24, 12, 17)  # Partition 10
]

def draw_partitions(image, partitions):
    if len(image.shape) == 2:
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    for (x_start, x_end, y_start, y_end) in partitions:
        cv2.rectangle(image, (x_start * SCALING_FACTOR, y_start * SCALING_FACTOR), 
                      (x_end * SCALING_FACTOR, y_end * SCALING_FACTOR), (0, 255, 255), 2)
    return image

def draw_hand_connections(image, landmarks):
    connections = [
        (0, 2), (2, 5), (5, 9), (9, 13), (13, 17),  # Palm
        (0, 1), (1, 2), (2, 3), (3, 4),  # 拇指
        (0, 5), (5, 6), (6, 7), (7, 8),  # 食指
        (0, 9), (9, 10), (10, 11), (11, 12),  # 中指
        (0, 13), (13, 14), (14, 15), (15, 16),  # 無名指
        (0, 17), (17, 18), (18, 19), (19, 20)  # 小指
    ]

    for start, end in connections:
        start_point = (int(landmarks[start][0]), int(landmarks[start][1]))
        end_point = (int(landmarks[end][0]), int(landmarks[end][1]))
        cv2.line(image, start_point, end_point, (0, 255, 0), 2)

class MainWindow(QtWidgets.QMainWindow):
    update_image_signal = QtCore.pyqtSignal(bytes)
    update_landmark_signal = QtCore.pyqtSignal(np.ndarray)
    update_motion_signal = QtCore.pyqtSignal(np.ndarray)

    def __init__(self):
        super().__init__()
        self.setWindowTitle("HandiMote Viewer V03")
        self.setGeometry(100, 100, 1200, 600)
        central_widget = QtWidgets.QWidget()
        self.setCentralWidget(central_widget)
        layout = QtWidgets.QVBoxLayout(central_widget)
        h_layout = QtWidgets.QHBoxLayout()
        layout.addLayout(h_layout)
        self.gray_image_label = QtWidgets.QLabel(self)
        self.gray_image_label.setFixedSize(NEW_SIZE[0], NEW_SIZE[1])
        h_layout.addWidget(self.gray_image_label)
        self.landmark_display_label = QtWidgets.QLabel(self)
        self.landmark_display_label.setFixedSize(640, 480)
        h_layout.addWidget(self.landmark_display_label)
        self.motion_display_label = QtWidgets.QLabel(self)
        self.motion_display_label.setFixedSize(640, 480)
        h_layout.addWidget(self.motion_display_label)
        self.quit_button = QtWidgets.QPushButton("Quit", self)
        self.quit_button.clicked.connect(self.close_app)
        layout.addWidget(self.quit_button)
        self.update_image_signal.connect(self.update_gray_image)
        self.update_landmark_signal.connect(self.update_landmark_image)
        self.update_motion_signal.connect(self.update_motion_image)

        device_address = self.load_device_address()
        print(f"Configured device address: {device_address}")

        self.reference_distance, ok = QtWidgets.QInputDialog.getDouble(self, 'Reference Distance', 'Enter wrist to thumb MCP distance (cm):')
        if not ok or self.reference_distance <= 0:
            self.reference_distance = 6.0

        self.timer = QtCore.QTimer(self)
        self.timer.timeout.connect(self.process_and_update_landmarks)
        self.timer.start(100)

        self.clear_path_timer = QtCore.QTimer(self)
        self.clear_path_timer.timeout.connect(self.clear_motion_path)
        self.clear_path_timer.start(2000)

        root = tk.Tk()
        root.withdraw()
        model_path = filedialog.askopenfilename(title="選擇訓練好的模型文件", filetypes=[("TFLite模型", "*.tflite")])

        if not model_path:
            print("模型文件未選擇。")
            sys.exit()

        try:
            self.interpreter = tf.lite.Interpreter(model_path=model_path)
            self.interpreter.allocate_tensors()
            self.input_details = self.interpreter.get_input_details()
            self.output_details = self.interpreter.get_output_details()
            print("模型已成功載入。")
        except Exception as e:
            print(f"載入模型時出錯: {e}")
            sys.exit()

        self.motion_path = []

    def load_device_address(self):
        try:
            with open('add.h', 'r') as file:
                address = file.read().strip()
                return address
        except Exception as e:
            print(f"Error reading device address: {e}")
            return None

    def handle_gray_image_data(self, image_data):
        self.update_image_signal.emit(bytes(image_data))

    def update_gray_image(self, image_data):
        global current_image_data
        np_image = np.frombuffer(image_data, dtype=np.uint8).reshape(20, 24)
        np_image = np_image[3:, :]
        resized_image = cv2.resize(np_image, (NEW_SIZE[0], NEW_SIZE[1]), interpolation=cv2.INTER_NEAREST)
        if len(resized_image.shape) == 3 and resized_image.shape[2] == 3:
            resized_image = cv2.cvtColor(resized_image, cv2.COLOR_BGR2GRAY)
        resized_image_with_partitions = draw_partitions(resized_image.copy(), combined_partitions)
        q_image = QtGui.QImage(resized_image_with_partitions.data, 
                               resized_image_with_partitions.shape[1], 
                               resized_image_with_partitions.shape[0], 
                               QtGui.QImage.Format_RGB888)
        pixmap = QtGui.QPixmap.fromImage(q_image)
        self.gray_image_label.setPixmap(pixmap)
        current_image_data = image_data

    def update_landmark_image(self, landmarks):
        blank_image = np.zeros((480, 640, 3), dtype=np.uint8)
        if landmarks.size > 0:
            scaled_landmarks = self.scale_landmarks(landmarks)
            display_scale_factor = 15
            scaled_landmarks *= display_scale_factor
            relative_landmarks = scaled_landmarks - scaled_landmarks[0]
            wrist_position = np.array([320, 360])
            translated_landmarks = relative_landmarks + np.array([wrist_position[0], wrist_position[1], 0])

            draw_hand_connections(blank_image, translated_landmarks)

            q_image = QtGui.QImage(blank_image.data, blank_image.shape[1], blank_image.shape[0], QtGui.QImage.Format_RGB888)
            pixmap = QtGui.QPixmap.fromImage(q_image)
            self.landmark_display_label.setPixmap(pixmap)

    def scale_landmarks(self, landmarks):
        wrist = landmarks[0]
        thumb_mcp = landmarks[2]
        actual_distance = np.linalg.norm(thumb_mcp - wrist)
        scale_factor = self.reference_distance / actual_distance
        scaled_landmarks = landmarks * scale_factor
        return scaled_landmarks

    def close_app(self):
        global client, ble_device
        try:
            if client and ble_device:
                asyncio.run_coroutine_threadsafe(shutdown(loop, client, ble_device), loop).result()
        except Exception as e:
            print(f"Error during shutdown: {e}")
        self.close()
        loop.call_soon_threadsafe(loop.stop)

    def process_and_update_landmarks(self):
        if current_image_data is not None:
            sensor_data = load_sensor_data(current_image_data)
            self.interpreter.set_tensor(self.input_details[0]['index'], sensor_data)
            self.interpreter.invoke()
            landmark_predictions = self.interpreter.get_tensor(self.output_details[0]['index'])
            landmark_predictions = landmark_predictions.reshape(-1, 3)

            self.landmark_data = landmark_predictions
            self.update_landmark_signal.emit(self.landmark_data)

    def update_motion_image(self, motion_data):
        try:
            qw, qx, qy, qz, _, _ = parse_motion_data(motion_data)
            if qw is None:
                return
            rotation_matrix = quaternion_to_rotation_matrix(qw, qx, qy, qz)

            if len(self.motion_path) == 0:
                self.motion_path.append([0, 0, 0])  # 中心起點

            last_point = self.motion_path[-1]

            # 使用旋轉矩陣更新軌跡位置
            new_point = np.dot(rotation_matrix, np.array([0, 0, 0])) + last_point
            self.motion_path.append(new_point)

            # 繪製運動軌跡
            blank_image = np.zeros((480, 640, 3), dtype=np.uint8)
            self.draw_motion_path(blank_image)
            self.draw_motion_cube(blank_image, new_point, rotation_matrix)
            
            q_image = QtGui.QImage(blank_image.data, blank_image.shape[1], blank_image.shape[0], QtGui.QImage.Format_RGB888)
            pixmap = QtGui.QPixmap.fromImage(q_image)
            self.motion_display_label.setPixmap(pixmap)
        except Exception as e:
            print(f"Error processing motion data: {e}")

    def draw_motion_path(self, image):
        for i in range(1, len(self.motion_path)):
            start_point = (int(self.motion_path[i-1][0]), int(self.motion_path[i-1][1]))
            end_point = (int(self.motion_path[i][0]), int(self.motion_path[i][1]))
            cv2.line(image, start_point, end_point, (0, 255, 0), 2)

    def draw_motion_cube(self, image, center, rotation_matrix):
        size = 100  # 放大立方體的大小
        points = np.array([
            [center[0] - size, center[1] - size, center[2] - size],
            [center[0] + size, center[1] - size, center[2] - size],
            [center[0] + size, center[1] + size, center[2] - size],
            [center[0] - size, center[1] + size, center[2] - size],
            [center[0] - size, center[1] - size, center[2] + size],
            [center[0] + size, center[1] - size, center[2] + size],
            [center[0] + size, center[1] + size, center[2] + size],
            [center[0] - size, center[1] + size, center[2] + size]
        ])
        points = np.dot(rotation_matrix, (points - center).T).T + center  # 旋轉立方體

        # 投影到2D平面
        def project_point(point):
            scale = 0.5  # 縮小座標顯示比例
            return (int(point[0] * scale + 320), int(point[1] * scale + 240))

        edges = [
            (0, 1), (1, 2), (2, 3), (3, 0),  # 底面
            (4, 5), (5, 6), (6, 7), (7, 4),  # 頂面
            (0, 4), (1, 5), (2, 6), (3, 7)   # 側面
        ]

        for start, end in edges:
            start_point = project_point(points[start])
            end_point = project_point(points[end])
            cv2.line(image, start_point, end_point, (255, 0, 0), 2)

    def clear_motion_path(self):
        self.motion_path = []

async def main(loop, window):
    device_found = False
    while not device_found:
        devices = await scan_for_devices("HandiMote")
        if devices:
            for device in devices:
                print(f"Found device: {device.address}")
            device_found = True
            try:
                client = await connect_to_device(devices[0])
                if client:
                    ble_device = BLEDevice(client)
                    if await ble_device.connect():
                        await ble_device.start_notifications(UUID_GRAY_LEVEL_IMAGE, data_collector.handle_gray_level_image_data)
                        await ble_device.start_notifications(UUID_6DOF_MOTION_SENSOR, data_collector.handle_motion_data)
                        while True:
                            await asyncio.sleep(0.1)
                        await ble_device.disconnect()
            except Exception as e:
                print(f"Error connecting to device: {e}")
        else:
            print("Device not found. Retrying in 5 seconds...")
            await asyncio.sleep(5)

async def shutdown(loop, client, ble_device):
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
    asyncio.set_event_loop(loop)
    try:
        loop.run_forever()
    finally:
        loop.close()

def main_function():
    global loop, app, window, cap
    app = QtWidgets.QApplication(sys.argv)
    window = MainWindow()
    window.show()
    loop = asyncio.new_event_loop()
    thread = threading.Thread(target=start_asyncio_loop)
    thread.start()
    asyncio.run_coroutine_threadsafe(main(loop, window), loop)
    cap = cv2.VideoCapture(0)
    sys.exit(app.exec_())

def load_sensor_data(image_data):
    sensor_data = np.frombuffer(image_data, dtype=np.uint8).reshape(20, 24)
    sensor_data = sensor_data[3:, :].astype('float32')
    sensor_data = sensor_data[np.newaxis, ..., np.newaxis]
    return sensor_data

if __name__ == "__main__":
    data_collector = DataCollector(update_image_callback=lambda data: window.handle_gray_image_data(data), 
                                   update_motion_callback=lambda data: window.update_motion_image(data))
    main_function()
