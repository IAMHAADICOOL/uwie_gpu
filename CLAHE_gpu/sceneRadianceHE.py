import cv2
import cupy as cp

def RecoverHE(sceneRadiance):
    # Convert image to CuPy array
    sceneRadiance = cp.asarray(sceneRadiance)

    # Apply Histogram Equalization to each channel
    for i in range(3):
        channel_np = cp.asnumpy(sceneRadiance[:, :, i])  # Convert to NumPy
        channel_np = cv2.equalizeHist(channel_np)  # Apply Histogram Equalization
        sceneRadiance[:, :, i] = cp.asarray(channel_np)  # Convert back to CuPy

    return sceneRadiance
