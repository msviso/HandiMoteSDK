# opencv_processor.py
import cv2
import numpy as np

def process_image(image_data, image_height, image_width):
    """
    This is a sample image processing function that can be modified according to your needs.
    
    Parameters:
    - image_data: The raw image data in bytes.
    - image_height: The height of the image.
    - image_width: The width of the image.
    
    Returns:
    - edges: The processed image after applying edge detection.
    """
    # Convert the byte data to a NumPy array and reshape it to the image dimensions
    np_image = np.frombuffer(image_data, dtype=np.uint8).reshape(image_height, image_width)
    
    # Assuming the data is already a grayscale image, no need to convert again
    gray_image = np_image
    
    # Perform edge detection using the Canny algorithm
    edges = cv2.Canny(gray_image, 100, 200)
    
    return edges

def match_templates(image_data, image_height, image_width, templates):
    """
    This function matches the input image against multiple template images and returns the match results.
    
    Parameters:
    - image_data: The raw image data in bytes.
    - image_height: The height of the image.
    - image_width: The width of the image.
    - templates: A list of file paths to the template images.
    
    Returns:
    - results: A list of tuples, each containing the template path, maximum match value, and location of the best match.
    """
    # Convert the byte data to a NumPy array and reshape it to the image dimensions
    np_image = np.frombuffer(image_data, dtype=np.uint8).reshape(image_height, image_width)
    
    results = []
    for template_path in templates:
        # Read the template image and convert it to a grayscale image
        template = cv2.imread(template_path, 0)
        
        if template is None:
            print(f"Template image {template_path} not found.")
            continue
        
        # Perform template matching using the normalized cross-correlation method
        result = cv2.matchTemplate(np_image, template, cv2.TM_CCOEFF_NORMED)
        
        # Get the minimum and maximum correlation values and their locations
        min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(result)
        
        # Append the template path, maximum correlation value, and location of the best match to the results list
        results.append((template_path, max_val, max_loc))
    
    return results


def extract_landmarks(image_path):
    image = cv2.imread(image_path)
    if image is None:
        print(f"Failed to load image from {image_path}")
        return None
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    result = hands.process(image_rgb)
    if result.multi_hand_landmarks:
        landmarks = result.multi_hand_landmarks[0]
        return np.array([[lm.x, lm.y, lm.z] for lm in landmarks.landmark]).flatten()
    else:
        return None
