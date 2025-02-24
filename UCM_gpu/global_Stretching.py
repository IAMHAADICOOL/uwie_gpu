import cupy as cp

def global_stretching(img_L, height, width):
    # Compute the min and max intensities using CuPy
    I_min = cp.min(img_L)
    I_max = cp.max(img_L)
    # Vectorized normalization to the range [0, 1]
    stretched = (img_L - I_min) * (1.0 / (I_max - I_min))
    return stretched
