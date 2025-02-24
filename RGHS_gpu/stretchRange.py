import cupy as cp
from scipy import stats

def stretchrange(r_array, height, width):
    r_array = cp.asarray(r_array).flatten()
    r_array = cp.sort(r_array)  # Sort values on GPU

    mode_np = stats.mode(cp.asnumpy(r_array)).mode[0]  # Convert to NumPy for mode computation
    mode = cp.asarray(mode_np)  # Convert mode back to CuPy for GPU operations

    mode_index_before = cp.where(r_array == mode)[0][0]

    SR_min = r_array[int(mode_index_before * 0.005)]
    SR_max = r_array[int(-(height * width - mode_index_before) * 0.005)]

    return SR_min, SR_max, mode
