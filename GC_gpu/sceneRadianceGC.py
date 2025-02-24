import cupy as cp

def RecoverGC(sceneRadiance):
    # Convert image to CuPy array and normalize to [0, 1]
    sceneRadiance = cp.asarray(sceneRadiance) / 255.0
    
    # Apply Gamma Correction (element-wise power operation on GPU)
    for i in range(3):
        max_val = cp.max(sceneRadiance[:, :, i])
        if max_val > 0:  # Avoid division by zero
            sceneRadiance[:, :, i] = cp.power(sceneRadiance[:, :, i] / max_val, 0.7)
    
    # Scale back to [0, 255], clip, and convert to uint8
    sceneRadiance = cp.clip(sceneRadiance * 255, 0, 255).astype(cp.uint8)
    
    return sceneRadiance
