import cupy as cp

def stretching(img):
    height, width = img.shape[:2]
    img = cp.asarray(img)  # Move to GPU

    for k in range(3):
        Max_channel = cp.max(img[:, :, k])
        Min_channel = cp.min(img[:, :, k])

        # Apply histogram stretching using vectorized operations on the GPU
        img[:, :, k] = (img[:, :, k] - Min_channel) * (255 / (Max_channel - Min_channel))

    return img
