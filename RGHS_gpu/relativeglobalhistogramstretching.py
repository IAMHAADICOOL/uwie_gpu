import cupy as cp
from stretchRange import stretchrange

def global_stretching(r_array, height, width, lamda, k):
    r_array = cp.asarray(r_array).flatten()
    r_array = cp.sort(r_array)

    I_min = r_array[int(height * width / 200)]
    I_max = r_array[-int(height * width / 200)]

    SR_min, SR_max, mode = stretchrange(r_array, height, width)
    DR_min = (1 - 0.655) * mode
    t_n = lamda ** 4
    O_max_left = SR_max * t_n * k / mode
    O_max_right = 255 * t_n * k / mode
    Dif = O_max_right - O_max_left

    if Dif >= 1:
        sum = cp.sum((1.526 + cp.arange(1, int(Dif) + 1)) * mode / (t_n * k))
        DR_max = sum / int(Dif)
    else:
        DR_max = mode

    r_array = cp.clip((r_array - I_min) * ((255 - I_min) / (I_max - I_min)), DR_min, DR_max)
    return r_array.reshape(height, width)

def RelativeGHstretching(sceneRadiance, height, width):
    sceneRadiance = cp.asarray(sceneRadiance)

    for i in range(3):
        sceneRadiance[:, :, i] = global_stretching(sceneRadiance[:, :, i], height, width, 0.97, 1.25)

    return sceneRadiance
