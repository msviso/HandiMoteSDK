import numpy as np
import struct
import time

def parse_motion_data(data):
    """
    Parses the motion data received from a sensor.
    
    Parameters:
    - data: A byte array containing the sensor data.
    
    Returns:
    - qw, qx, qy, qz: The quaternion components.
    - current_time: The current time as a float.
    - event: The event type or code.
    """
    if len(data) >= 21:  # Ensure there's enough data for the new format
        # Unpack the first 21 bytes of data into the quaternion components, timestamp, and event code
        qx, qy, qz, qw, timestamp, event = struct.unpack('<4fIB', data[:21])
        # Return the quaternion components, the current time, and the event code
        return qw, qx, qy, qz, time.time(), event  
    # Default values in case of insufficient data
    return None, None, None, None, None, None

def quaternion_to_rotation_matrix(qw, qx, qy, qz):
    """
    Converts a quaternion into a rotation matrix.
    
    Parameters:
    - qw, qx, qy, qz: The quaternion components.
    
    Returns:
    - rotation_matrix: A 3x3 NumPy array representing the rotation matrix.
    """
    # Compute the elements of the rotation matrix from the quaternion components
    r11 = 1 - 2 * (qy**2 + qz**2)
    r12 = 2 * (qx*qy - qz*qw)
    r13 = 2 * (qx*qz + qy*qw)
    r21 = 2 * (qx*qy + qz*qw)
    r22 = 1 - 2 * (qx**2 + qz**2)
    r23 = 2 * (qy*qz - qx*qw)
    r31 = 2 * (qx*qz - qy*qw)
    r32 = 2 * (qy*qz + qx*qw)
    r33 = 1 - 2 * (qx**2 + qy**2)
    
    # Create the rotation matrix
    rotation_matrix = np.array([
        [r11, r12, r13],
        [r21, r22, r23],
        [r31, r32, r33]
    ])
    return rotation_matrix

def quaternion_multiply(q1, q2):
    """
    Multiplies two quaternions.
    
    Parameters:
    - q1, q2: Tuples representing the quaternions to be multiplied.
    
    Returns:
    - w, x, y, z: The resulting quaternion components after multiplication.
    """
    w1, x1, y1, z1 = q1
    w2, x2, y2, z2 = q2
    
    # Perform the quaternion multiplication
    w = w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2
    x = w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2
    y = w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2
    z = w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2
    
    return w, x, y, z

def quaternion_conjugate(quaternion):
    """
    Computes the conjugate of a quaternion.
    
    Parameters:
    - quaternion: A tuple representing the quaternion.
    
    Returns:
    - The conjugate of the quaternion as a tuple.
    """
    w, x, y, z = quaternion
    return (w, -x, -y, -z)
