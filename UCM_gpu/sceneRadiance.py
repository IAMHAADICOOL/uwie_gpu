import cupy as cp

def sceneRadianceRGB(sceneRadiance):
    # Clip the scene radiance values and convert to uint8 on the GPU
    sceneRadiance = cp.clip(sceneRadiance, 0, 255)
    sceneRadiance = sceneRadiance.astype(cp.uint8)
    # Convert the final result back to a NumPy array for saving with cv2
    return cp.asnumpy(sceneRadiance)
