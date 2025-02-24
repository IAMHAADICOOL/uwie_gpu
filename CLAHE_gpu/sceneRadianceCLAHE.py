import cv2
import cupy as cp

def RecoverCLAHE(sceneRadiance):
    # Convert image to CuPy array
    sceneRadiance = cp.asarray(sceneRadiance)
    
    # Create CLAHE object
    clahe = cv2.createCLAHE(clipLimit=2, tileGridSize=(4, 4))
    
    # Apply CLAHE to each channel
    for i in range(3):
        channel_np = cp.asnumpy(sceneRadiance[:, :, i])  # Convert to NumPy for OpenCV
        channel_np = clahe.apply(channel_np)  # Apply CLAHE
        sceneRadiance[:, :, i] = cp.asarray(channel_np)  # Convert back to CuPy

    return sceneRadiance
