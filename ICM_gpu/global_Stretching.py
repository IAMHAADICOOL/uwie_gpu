import cupy as cp

def global_stretching(img_L, height, width):
    img_L = cp.asarray(img_L)  # Move to GPU

    I_min = cp.min(img_L)
    I_max = cp.max(img_L)

    # Normalize using vectorized operations
    array_Global_histogram_stretching_L = (img_L - I_min) / (I_max - I_min)

    return array_Global_histogram_stretching_L
