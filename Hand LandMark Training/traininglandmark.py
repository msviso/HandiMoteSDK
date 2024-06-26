import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Input, BatchNormalization, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import tkinter as tk
from tkinter import filedialog

# 定義模型結構
def create_model(input_shape):
    model = Sequential()
    model.add(Input(shape=input_shape))
    model.add(Conv2D(32, (3, 3), activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.01)))
    model.add(BatchNormalization())
    model.add(MaxPooling2D((2, 2)))
    model.add(Dropout(0.25))

    model.add(Conv2D(64, (3, 3), activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.01)))
    model.add(BatchNormalization())
    model.add(MaxPooling2D((2, 2)))
    model.add(Dropout(0.25))

    model.add(Flatten())
    model.add(Dense(128, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.01)))
    model.add(BatchNormalization())
    model.add(Dropout(0.5))

    model.add(Dense(63))  # 輸出層有 63 個單元（21 個關鍵點 * 3 個座標）
    return model

# 從文件中載入感測器和標記數據
def load_data(sensor_dir, landmark_dir):
    sensor_data = []
    landmark_data = []

    for file in os.listdir(sensor_dir):
        if file.endswith(".bin"):
            sensor_file = os.path.join(sensor_dir, file)
            landmark_file = os.path.join(landmark_dir, file.replace('vision_data', 'landmark_data').replace('.bin', '.csv'))

            if not os.path.isfile(sensor_file):
                print(f"感測器文件未找到: {sensor_file}")
                continue
            if not os.path.isfile(landmark_file):
                print(f"標記文件未找到: {landmark_file}")
                continue

            sensor = np.fromfile(sensor_file, dtype=np.uint8).reshape(20, 24)  # 讀取並重塑二進位數據
            landmark = np.loadtxt(landmark_file, delimiter=',', skiprows=1)

            # 僅使用有效區域 (17x24)
            sensor = sensor[3:, :].astype('float32') / 255.0  # 將數據轉換為 float32 並標準化
            landmark = landmark.flatten().astype('float32')  # 標準化landmark數據

            # 打印landmark的形狀以進行確認
            print(f"Loaded landmark data with shape: {landmark.shape}")

            sensor_data.append(sensor)
            landmark_data.append(landmark)

    if not sensor_data or not landmark_data:
        raise ValueError("未找到有效的訓練數據。")

    return np.array(sensor_data), np.array(landmark_data)

# 主訓練函數
def main():
    # 請求使用者選擇根目錄
    root = tk.Tk()
    root.withdraw()
    root_dir = filedialog.askdirectory(title="選擇包含 'vision' 和 'landmark' 文件夾的根目錄")
    sensor_dir = os.path.join(root_dir, "vision")
    landmark_dir = os.path.join(root_dir, "landmark")

    sensor_data, landmark_data = load_data(sensor_dir, landmark_dir)

    # 將數據調整為 (17, 24, 1) 的形狀
    sensor_data = sensor_data[..., np.newaxis]

    # 將數據拆分為訓練和測試集
    split_index = int(0.8 * len(sensor_data))
    X_train, X_test = sensor_data[:split_index], sensor_data[split_index:]
    y_train, y_test = landmark_data[:split_index], landmark_data[split_index:]

    # 數據增強
    datagen = ImageDataGenerator(
        rotation_range=10,
        width_shift_range=0.1,
        height_shift_range=0.1,
        zoom_range=0.1
    )

    # 載入或創建模型
    model_path = '1hand_landmark_model.keras'
    if os.path.exists(model_path):
        try:
            model = load_model(model_path, compile=False)
            model.compile(optimizer=Adam(learning_rate=0.001), loss='mean_squared_error')
            print("已載入現有模型和權重。")
        except Exception as e:
            print(f"載入模型時出錯: {e}")
            model = create_model((17, 24, 1))
            model.compile(optimizer=Adam(learning_rate=0.001), loss='mean_squared_error')
            print("訓練新模型。")
    else:
        model = create_model((17, 24, 1))
        model.compile(optimizer=Adam(learning_rate=0.001), loss='mean_squared_error')
        print("訓練新模型。")

    # 訓練模型
    model.fit(datagen.flow(X_train, y_train, batch_size=32), epochs=50, validation_data=(X_test, y_test))

    # 保存模型
    model.save(model_path)
    print("模型保存至", model_path)

if __name__ == "__main__":
    main()
