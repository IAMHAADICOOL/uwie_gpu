import cv2
from skimage.color import rgb2hsv, hsv2rgb
import cupy as cp
import numpy as np
from global_Stretching import global_stretching

def HSVStretching(sceneRadiance):
    # Convert to NumPy for color space conversion
    sceneRadiance_np = cp.asnumpy(sceneRadiance)
    img_hsv = rgb2hsv(sceneRadiance_np)

    h, s, v = cv2.split(img_hsv)

    # Move s and v channels to GPU
    s_gpu = cp.asarray(s)
    v_gpu = cp.asarray(v)

    # Apply global stretching on the GPU
    img_s_stretching = global_stretching(s_gpu, *s.shape)
    img_v_stretching = global_stretching(v_gpu, *v.shape)

    # Convert back to NumPy for conversion back to RGB
    img_s_stretching_np = cp.asnumpy(img_s_stretching)
    img_v_stretching_np = cp.asnumpy(img_v_stretching)

    # Reconstruct the HSV image
    labArray = np.zeros((*sceneRadiance.shape[:2], 3), dtype=np.float64)
    labArray[:, :, 0] = h
    labArray[:, :, 1] = img_s_stretching_np
    labArray[:, :, 2] = img_v_stretching_np

    # Convert back to RGB
    img_rgb = hsv2rgb(labArray) * 255

    return img_rgb
