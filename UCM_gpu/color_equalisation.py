import cupy as cp

def cal_equalisation(img, ratio):
    # Multiply by ratio and clip the result to [0, 255]
    Array = img * ratio
    Array = cp.clip(Array, 0, 255)
    return Array

def RGB_equalisation(img):
    # Ensure the image is a CuPy array in float32
    img = cp.asarray(img).astype(cp.float32)
    avg_RGB = []
    for i in range(3):
        avg = cp.mean(img[:, :, i])
        avg_RGB.append(avg)
    # Compute scaling ratios for the red and green channels based on the blue channel
    a_r = avg_RGB[0] / avg_RGB[2]
    a_g = avg_RGB[0] / avg_RGB[1]
    ratio = [0, a_g, a_r]
    for i in range(1, 3):
        img[:, :, i] = cal_equalisation(img[:, :, i], ratio[i])
    return img
