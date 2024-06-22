import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
import tkinter as tk
from tkinter import filedialog
import matplotlib.pyplot as plt

# 定義函數來載入感測器數據
def load_sensor_data(file_path):
    sensor_data = np.fromfile(file_path, dtype=np.uint8).reshape(20, 24)  # 讀取並重塑二進位數據
    sensor_data = sensor_data[3:, :].astype('float32')  # 僅使用有效區域 (17x24)，並將數據轉換為 float32
    sensor_data = sensor_data[np.newaxis, ..., np.newaxis]  # 增加維度以符合模型輸入格式 (1, 17, 24, 1)
    return sensor_data

# 繪製landmark的函數
def plot_landmarks(landmarks):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_title('3D Hand Landmarks')

    # 繪製每個landmark點
    for (x, y, z) in landmarks:
        ax.scatter(x, y, z, c='r', marker='o')
    
    # 繪製MCP線段
    mcp_indices = [0, 5, 9, 13, 17, 0]  # MCP 點的索引
    for i in range(len(mcp_indices) - 1):
        x_vals = [landmarks[mcp_indices[i]][0], landmarks[mcp_indices[i+1]][0]]
        y_vals = [landmarks[mcp_indices[i]][1], landmarks[mcp_indices[i+1]][1]]
        z_vals = [landmarks[mcp_indices[i]][2], landmarks[mcp_indices[i+1]][2]]
        ax.plot(x_vals, y_vals, z_vals, 'b-')

    # 定義每個手指的關節點索引
    finger_indices = [
        [0, 1, 2, 3, 4],  # 拇指
        [0, 5, 6, 7, 8],  # 食指
        [9, 10, 11, 12],  # 中指
        [13, 14, 15, 16],  # 無名指
        [17, 18, 19, 20]  # 小指
    ]

    # 繪製每個手指的關節線段
    for finger in finger_indices:
        for i in range(len(finger) - 1):
            x_vals = [landmarks[finger[i]][0], landmarks[finger[i+1]][0]]
            y_vals = [landmarks[finger[i]][1], landmarks[finger[i+1]][1]]
            z_vals = [landmarks[finger[i]][2], landmarks[finger[i+1]][2]]
            ax.plot(x_vals, y_vals, z_vals, 'g-')

    plt.show()

# 主函數
def main():
    # 請求使用者選擇模型文件
    root = tk.Tk()
    root.withdraw()
    model_path = filedialog.askopenfilename(title="選擇訓練好的模型文件", filetypes=[("Keras模型", "*.keras")])

    if not model_path:
        print("模型文件未選擇。")
        return

    # 載入模型
    try:
        model = load_model(model_path, compile=False)
        model.compile(optimizer='adam', loss='mean_squared_error')
        print("模型已成功載入。")
    except Exception as e:
        print(f"載入模型時出錯: {e}")
        return

    while True:
        # 請求使用者選擇vision的bin文件
        vision_file_path = filedialog.askopenfilename(title="選擇vision的bin文件", filetypes=[("二進位文件", "*.bin")])

        if not vision_file_path:
            print("vision文件未選擇。")
            break

        # 載入感測器數據
        sensor_data = load_sensor_data(vision_file_path)

        # 使用模型進行預測
        landmark_predictions = model.predict(sensor_data)
        landmark_predictions = landmark_predictions.reshape(-1, 3)  # 將預測結果重塑為 (21, 3)

        # 顯示結果
        print("Landmark 預測結果 (x, y, z):")
        for i, (x, y, z) in enumerate(landmark_predictions):
            print(f"點 {i+1}: x={x:.4f}, y={y:.4f}, z={z:.4f}")

        # 繪製landmark
        plot_landmarks(landmark_predictions)

if __name__ == "__main__":
    main()
