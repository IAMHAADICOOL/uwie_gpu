import cv2
from skimage.color import rgb2lab, lab2rgb
import cupy as cp
import numpy as np
from global_StretchingL import global_stretching
from global_stretching_ab import global_Stretching_ab

def LABStretching(sceneRadiance):
    sceneRadiance = cp.asarray(sceneRadiance)
    sceneRadiance = cp.clip(sceneRadiance, 0, 255).astype(cp.uint8)

    # Convert to NumPy for color conversion
    sceneRadiance_np = cp.asnumpy(sceneRadiance)
    img_lab = rgb2lab(sceneRadiance_np)
    L, a, b = cv2.split(img_lab)

    # Move to GPU
    L_gpu, a_gpu, b_gpu = cp.asarray(L), cp.asarray(a), cp.asarray(b)

    img_L_stretching = global_stretching(L_gpu, *sceneRadiance.shape[:2])
    img_a_stretching = global_Stretching_ab(a_gpu, *sceneRadiance.shape[:2])
    img_b_stretching = global_Stretching_ab(b_gpu, *sceneRadiance.shape[:2])

    # Convert back to NumPy for final conversion
    labArray = np.zeros((*sceneRadiance.shape[:2], 3), dtype=np.float64)
    labArray[:, :, 0] = cp.asnumpy(img_L_stretching)
    labArray[:, :, 1] = cp.asnumpy(img_a_stretching)
    labArray[:, :, 2] = cp.asnumpy(img_b_stretching)

    img_rgb = lab2rgb(labArray) * 255
    return img_rgb
