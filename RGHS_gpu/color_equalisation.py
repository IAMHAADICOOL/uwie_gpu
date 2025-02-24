import cupy as cp

def cal_equalisation(img, ratio):
    Array = img * ratio
    Array = cp.clip(Array, 0, 255)
    return Array

def RGB_equalisation(img):
    img = cp.asarray(img).astype(cp.float32)
    avg_RGB = cp.mean(img, axis=(0, 1))
    ratio = 128 / avg_RGB

    for i in range(3):  # Apply equalization to all three channels
        img[:, :, i] = cal_equalisation(img[:, :, i], ratio[i])

    return img
