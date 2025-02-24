import cv2
from skimage.color import rgb2hsv, hsv2rgb
import cupy as cp
import numpy as np
from global_Stretching import global_stretching

def HSVStretching(sceneRadiance):
    # Convert the input image to a CuPy array and ensure it is uint8
    sceneRadiance = cp.asarray(sceneRadiance).astype(cp.uint8)
    height, width = sceneRadiance.shape[:2]
    # Convert to NumPy for color space conversion (rgb2hsv requires NumPy arrays)
    sceneRadiance_np = cp.asnumpy(sceneRadiance)
    img_hsv = rgb2hsv(sceneRadiance_np)
    h, s, v = cv2.split(img_hsv)
    # Move the saturation and value channels to the GPU
    s_gpu = cp.asarray(s)
    v_gpu = cp.asarray(v)
    # Apply global stretching on the saturation and value channels using CuPy
    img_s_stretching = global_stretching(s_gpu, height, width)
    img_v_stretching = global_stretching(v_gpu, height, width)
    # Convert the results back to NumPy arrays
    img_s_stretching_np = cp.asnumpy(img_s_stretching)
    img_v_stretching_np = cp.asnumpy(img_v_stretching)
    # Reconstruct the HSV image and convert back to RGB
    labArray = np.zeros((height, width, 3), dtype=np.float64)
    labArray[:, :, 0] = h
    labArray[:, :, 1] = img_s_stretching_np
    labArray[:, :, 2] = img_v_stretching_np
    img_rgb = hsv2rgb(labArray) * 255
    return img_rgb
