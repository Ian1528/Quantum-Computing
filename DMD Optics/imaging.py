import os
from pymba import Vimba
import numpy as np
from scipy import ndimage
import matplotlib.pyplot as plt

def take_single_shot(exposure_time):
    """
    Takes and returns a single image with the given exposure time

    Args:
        exposure_time (int): exposure time in microseconds

    Returns:
        img: the image, which is a numpy array
    """
    with Vimba() as vimba:
        camera_ids = vimba.camera_ids()
        cam =  vimba.camera(camera_ids[0])
        cam.open()
        cam.arm(mode='SingleFrame')
        cam.ExposureTime = exposure_time
        frame = cam.acquire_frame(5000)
        image = frame.buffer_data_numpy().copy()
        cam.disarm()
        cam.close()
    return image
def take_multiple_exposures(exposure_times, destination=None):
    """
    Takes and returns a list of images, each taken with a different exposure time.

    Args:
        exposure_times (list of int): List of exposure times in microseconds.
        destination (str, optional): Directory to save images. If None, images are not saved.

    Returns:
        images (list of np.ndarray): List of captured images as numpy arrays.
    """
    if destination and not os.path.exists(destination):
        os.makedirs(destination)

    images = []
    with Vimba() as vimba:
        camera_ids = vimba.camera_ids()
        cam =  vimba.camera(camera_ids[0])
        cam.open()
        for t in exposure_times:
            cam.arm(mode='SingleFrame')

            cam.ExposureTime = t
            
            frame = cam.acquire_frame(10000)
            image = frame.buffer_data_numpy().copy()

            cam.disarm()
            images.append(image)

            if destination:
                np.save(os.path.join(destination, f"t_{t}"), image)
            print(f"Exposure time {t} completed. Maximum pixel value: {np.max(image)}")
    return images