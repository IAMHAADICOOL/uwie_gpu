import cupy as cp

def global_Stretching_ab(a, height, width):
    a = cp.asarray(a).astype(cp.float64)
    p_out = a * (1.3 ** (1 - cp.abs(a / 128)))
    return p_out
