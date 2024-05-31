# opencv_processor.py
import cv2
import numpy as np

def process_image(image_data, image_height, image_width):
    # 這是一個示例圖像處理函數，可以根據您的需求進行修改
    np_image = np.frombuffer(image_data, dtype=np.uint8).reshape(image_height, image_width)
    # 假設數據已經是灰度圖像，不需要再次轉換
    gray_image = np_image
    # 進行其他處理，例如邊緣檢測
    edges = cv2.Canny(gray_image, 100, 200)
    return edges

def match_templates(image_data, image_height, image_width, templates):
    np_image = np.frombuffer(image_data, dtype=np.uint8).reshape(image_height, image_width)
    results = []
    for template_path in templates:
        template = cv2.imread(template_path, 0)  # 讀取模板圖像並轉換為灰度圖像
        if template is None:
            print(f"Template image {template_path} not found.")
            continue

        result = cv2.matchTemplate(np_image, template, cv2.TM_CCOEFF_NORMED)
        min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(result)
        results.append((template_path, max_val, max_loc))
    return results