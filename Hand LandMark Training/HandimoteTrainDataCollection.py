import asyncio
import numpy as np
import threading
import mediapipe as mp
import cv2
from datetime import datetime
from PyQt5 import QtWidgets, QtGui, QtCore
import os
import sys
from utilities import scan_for_devices, connect_to_device, DataCollector
from ble_device import BLEDevice
from config import UUID_GRAY_LEVEL_IMAGE, UUID_6DOF_MOTION_SENSOR
from motion_processor import parse_motion_data

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
batch_name = None  # 批次名稱
reference_distance = 6.0  # 參考距離，設定為6cm

# Initialize MediaPipe
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=1,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.5
)

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
    # 定義手部連接的關係，從手掌到手指
    connections = [
        (0, 2), (2, 5), (5, 9), (9, 13), (13, 17), # Palm
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

    def __init__(self):
        super().__init__()
        self.setWindowTitle("HandiMote Viewer")
        self.setGeometry(100, 100, 1200, 600)  # 增加寬度以顯示完整地標座標
        central_widget = QtWidgets.QWidget()
        self.setCentralWidget(central_widget)
        layout = QtWidgets.QVBoxLayout(central_widget)
        h_layout = QtWidgets.QHBoxLayout()
        layout.addLayout(h_layout)
        self.gray_image_label = QtWidgets.QLabel(self)
        self.gray_image_label.setFixedSize(NEW_SIZE[0], NEW_SIZE[1])
        h_layout.addWidget(self.gray_image_label)
        self.landmark_coords_label = QtWidgets.QTextEdit(self)
        self.landmark_coords_label.setReadOnly(True)
        self.landmark_coords_label.setFixedSize(300, 480)  # 增加大小以顯示完整地標座標
        h_layout.addWidget(self.landmark_coords_label)
        self.landmark_display_label = QtWidgets.QLabel(self)
        self.landmark_display_label.setFixedSize(640, 480)  # 用於顯示Landmark的視窗
        h_layout.addWidget(self.landmark_display_label)
        self.capture_button = QtWidgets.QPushButton("Capture Data", self)
        self.capture_button.clicked.connect(self.capture_data)
        layout.addWidget(self.capture_button)
        self.quit_button = QtWidgets.QPushButton("Quit", self)
        self.quit_button.clicked.connect(self.close_app)
        layout.addWidget(self.quit_button)
        self.update_image_signal.connect(self.update_gray_image)
        self.update_landmark_signal.connect(self.update_landmark_image)

        # Load device address from add.h and display
        device_address = self.load_device_address()
        print(f"Configured device address: {device_address}")

        # 請求輸入批次名稱
        self.batch_name, ok = QtWidgets.QInputDialog.getText(self, 'Batch Name', 'Enter batch name:')
        if not ok or not self.batch_name:
            self.batch_name = datetime.now().strftime("%Y%m%d_%H%M%S")
        print(f"Using batch name: {self.batch_name}")

        # 請求輸入手腕到拇指MCP的距離
        self.reference_distance, ok = QtWidgets.QInputDialog.getDouble(self, 'Reference Distance', 'Enter wrist to thumb MCP distance (cm):')
        if not ok or self.reference_distance <= 0:
            self.reference_distance = 6.0  # 使用預設值6 cm

        # Start timer for updating landmark image
        self.timer = QtCore.QTimer(self)
        self.timer.timeout.connect(self.process_and_update_landmarks)
        self.timer.start(100)

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
        landmark_texts = []
        if landmarks.size > 0:
            scaled_landmarks = self.scale_landmarks(landmarks)
            # 放大顯示的座標
            display_scale_factor = 15  # 您可以根據需要調整這個值以放大顯示
            scaled_landmarks *= display_scale_factor
            relative_landmarks = scaled_landmarks - scaled_landmarks[0]  # 轉換為相對座標
            wrist_position = np.array([320, 360])  # 中心點偏下的位置
            translated_landmarks = relative_landmarks + np.array([wrist_position[0], wrist_position[1], 0])

            # 繪製連接線
            draw_hand_connections(blank_image, translated_landmarks)

            for idx, lm in enumerate(translated_landmarks):
                cx, cy = int(lm[0]), int(lm[1])
                # 動態調整地標點的大小
                radius = 5  # 您可以根據需要調整這個值
                cv2.circle(blank_image, (cx, cy), radius, (0, 255, 0), -1)
                landmark_texts.append(f"Point {idx}: ({lm[0]:.2f}, {lm[1]:.2f}, {lm[2]:.2f})")
            self.landmark_coords_label.setText("\n".join(landmark_texts))
            q_image = QtGui.QImage(blank_image.data, blank_image.shape[1], blank_image.shape[0], QtGui.QImage.Format_RGB888)
            pixmap = QtGui.QPixmap.fromImage(q_image)
            self.landmark_display_label.setPixmap(pixmap)

    def scale_landmarks(self, landmarks):
        wrist = landmarks[0]
        thumb_mcp = landmarks[2]  # 拇指 MCP
        actual_distance = np.linalg.norm(thumb_mcp - wrist)
        scale_factor = self.reference_distance / actual_distance
        scaled_landmarks = landmarks * scale_factor
        return scaled_landmarks

    def capture_data(self):
        global current_image_data, motion_data_list, landmark_data
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        batch_folder = self.batch_name
        vision_folder = os.path.join(batch_folder, "vision")
        motion_folder = os.path.join(batch_folder, "motion")
        landmark_folder = os.path.join(batch_folder, "landmark")

        # 創建資料夾
        for folder in [vision_folder, motion_folder, landmark_folder]:
            if not os.path.exists(folder):
                os.makedirs(folder)

        bin_filename = os.path.join(vision_folder, f"vision_data_{timestamp}_{self.batch_name}.bin")
        landmark_filename = os.path.join(landmark_folder, f"landmark_data_{timestamp}_{self.batch_name}.csv")
        motion_filename = os.path.join(motion_folder, f"motion_data_{timestamp}_{self.batch_name}.csv")

        try:
            if current_image_data is not None:
                with open(bin_filename, 'wb') as bin_file:
                    bin_file.write(current_image_data)
            if self.landmark_data is not None and self.landmark_data.size > 0:
                # 在存儲之前將 wrist 設置為 (0, 0, 0)
                aligned_landmarks = self.landmark_data - self.landmark_data[0]
                np.savetxt(landmark_filename, aligned_landmarks, delimiter=",", header="x,y,z", comments='')
            if motion_data_list:
                np.savetxt(motion_filename, motion_data_list, delimiter=",", header="timestamp,x,y,z", comments='')
            print(f"Captured data saved as {bin_filename}, {landmark_filename}, {motion_filename}")
        except Exception as e:
            print(f"Error saving captured data: {e}")

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
        success, image = cap.read()
        if success:
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            results = hands.process(image_rgb)

            if results.multi_hand_landmarks:
                for hand_landmarks in results.multi_hand_landmarks:
                    landmarks = np.array([[lm.x, lm.y, lm.z] for lm in hand_landmarks.landmark])
                    self.landmark_data = self.scale_landmarks(landmarks)
                    self.update_landmark_signal.emit(self.landmark_data)

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

if __name__ == "__main__":
    data_collector = DataCollector(update_image_callback=lambda data: window.handle_gray_image_data(data), 
                                   update_motion_callback=lambda data: motion_data_list.append(parse_motion_data(data)))
    main_function()
