import cupy as cp

def histogram_r(r_array, height, width):
    # Flatten and sort the channel values on the GPU
    flat = cp.sort(r_array.flatten())
    length = height * width
    I_min = flat[int(length / 500)]
    I_max = flat[-int(length / 500)]
    # Apply the stretching using vectorized operations
    stretched = cp.where(r_array < I_min, I_min,
                  cp.where(r_array > I_max, 255,
                  ((r_array - I_min) * ((255 - I_min) / (I_max - I_min)) + I_min).astype(cp.int32)))
    return stretched

def histogram_g(r_array, height, width):
    flat = cp.sort(r_array.flatten())
    length = height * width
    I_min = flat[int(length / 500)]
    I_max = flat[-int(length / 500)]
    stretched = cp.where(r_array < I_min, 0,
                  cp.where(r_array > I_max, 255,
                  ((r_array - I_min) * (255 / (I_max - I_min))).astype(cp.int32)))
    return stretched

def histogram_b(r_array, height, width):
    flat = cp.sort(r_array.flatten())
    length = height * width
    I_min = flat[int(length / 500)]
    I_max = flat[-int(length / 500)]
    stretched = cp.where(r_array < I_min, 0,
                  cp.where(r_array > I_max, I_max,
                  ((r_array - I_min) * (I_max / (I_max - I_min))).astype(cp.int32)))
    return stretched

def stretching(img):
    height, width = img.shape[:2]
    # Process each color channel using the corresponding histogram function
    img[:, :, 2] = histogram_r(img[:, :, 2], height, width)
    img[:, :, 1] = histogram_g(img[:, :, 1], height, width)
    img[:, :, 0] = histogram_b(img[:, :, 0], height, width)
    return img
