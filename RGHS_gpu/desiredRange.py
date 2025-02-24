import cupy as cp
from scipy import stats

def stretchrange(r_array, height, width):
    r_array = cp.asarray(r_array).flatten()
    r_array = cp.sort(r_array)  # Sort the values on GPU

    mode = cp.asnumpy(stats.mode(r_array).mode[0])  # Convert to NumPy for mode calculation
    mode_index_before = cp.asnumpy(cp.where(r_array == mode)[0][0])

    DR_min = (1 - 0.655) * mode
    SR_max = r_array[int(-(height * width - mode_index_before) * 0.005)]

    return DR_min, SR_max, mode
