# HandiMote Viewer

`HandiMote Viewer` is an application designed to display and analyze motion and image data received from the HandiMote device. This application includes three main features: OpenCV image processing, motion cube display, and motion tracking path and history.

![image](Images\HandiMoteViewerConsole.png)

## Code Structure
HandiMote Viewer/

├── add.h # Device address file

├── ble_device.py # BLE device handling code

├── config.py # Configuration file, including UUIDs and other settings

├── example_usage.py # Main application code

├── motion_processor.py # Code for processing motion data

├── notification_handler.py# Code for handling notifications

├── opencv_processor.py # Image processing code

├── setup.py # Setup and dependency installation

└── utilities.py # Utility functions and helper code




## Quick Setup Guide

1. **Clone the Repository**

    ```sh
    git clone https://github.com/yourusername/HandiMoteViewer.git
    cd HandiMoteViewer/HandiMoteSDK
    ```

2. **Install Dependencies**

    Ensure you have Python 3.8 or higher installed, then run the following command to install the necessary dependencies:

    ```sh
    pip install -r requirements.txt
    ```

3. **Configure Device Address**

    Edit the `add.h` file to set the BLE address of your HandiMote device:

    ```c
    #define DEVICE_ADDRESS "00:11:22:33:44:55"
    ```

4. **Run the Application**

    Run the main application:

    ```sh
    python example_usage.py
    ```

## Application Description

### OpenCV Image Processing

The application receives grayscale image data from the HandiMote device and processes it using OpenCV. The processed image is displayed in the GUI, and template matching is performed to recognize specific events. The matching results are displayed in a dedicated message box.

### Motion Cube Display

The application receives quaternion data from the HandiMote device and converts it into a rotation matrix to display the rotation of a motion cube. The cube's rotation is updated in real-time based on the device's motion.

### Motion Tracking Path and History

The application records the motion path of the device and displays these paths in 3D space. The motion path is updated in real-time based on the device's motion, showing the movement history of the last 3 seconds.

### Image Saving for OpenCV Training

The application allows users to capture the current grayscale image and save it as both a binary file and a PNG file. This feature is useful for collecting data for OpenCV training. When the capture button is pressed, the image is saved in the `captured_images` folder with a timestamp.

## Contribution

Contributions to this project are welcome! If you have any suggestions or find any issues, please submit an issue or a pull request.

## License

This project is licensed under the MIT License. For more details, please see the [LICENSE](LICENSE) file.
