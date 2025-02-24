import cupy as cp

def global_stretching(img_L, height, width):
    img_L = cp.asarray(img_L)
    length = height * width
    R_rray = cp.sort(img_L.flatten())  # Sort on GPU

    I_min = int(R_rray[int(length / 100)])
    I_max = int(R_rray[-int(length / 100)])

    img_L = cp.clip((img_L - I_min) * (100 / (I_max - I_min)), 0, 100)
    return img_L
