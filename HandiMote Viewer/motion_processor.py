# motion_processor.py
import numpy as np

def parse_motion_data(data):
    # 假設數據的格式為：四個 float 值和一個 uint32_t 時間戳和一個 uint8_t 事件
    qw, qx, qy, qz = np.frombuffer(data[:16], dtype=np.float32)
    timestamp = np.frombuffer(data[16:20], dtype=np.uint32)[0]
    event = data[20]
    return qw, qx, qy, qz, timestamp, event

def quaternion_to_rotation_matrix(qw, qx, qy, qz):
    return np.array([
        [1 - 2*qy*qy - 2*qz*qz, 2*qx*qy - 2*qz*qw, 2*qx*qz + 2*qy*qw],
        [2*qx*qy + 2*qz*qw, 1 - 2*qx*qx - 2*qz*qz, 2*qy*qz - 2*qx*qw],
        [2*qx*qz - 2*qy*qw, 2*qy*qz + 2*qx*qw, 1 - 2*qx*qx - 2*qy*qy]
    ])
