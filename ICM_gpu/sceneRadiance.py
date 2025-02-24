import cupy as cp

def sceneRadianceRGB(sceneRadiance):
    # Clip values and ensure 8-bit precision
    sceneRadiance = cp.clip(sceneRadiance, 0, 255).astype(cp.uint8)

    # Convert back to NumPy for OpenCV compatibility
    return cp.asnumpy(sceneRadiance)
