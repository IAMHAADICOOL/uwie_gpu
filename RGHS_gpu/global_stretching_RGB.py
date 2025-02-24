import cupy as cp

def stretching(img):
    img = cp.asarray(img)
    height, width = img.shape[:2]

    for k in range(3):
        Max_channel = cp.max(img[:, :, k])
        Min_channel = cp.min(img[:, :, k])
        img[:, :, k] = (img[:, :, k] - Min_channel) * (255 / (Max_channel - Min_channel))

    return img
