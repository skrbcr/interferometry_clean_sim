import numpy as np
import matplotlib.pyplot as plt
from casatools import ms as mstool  # CASA ms tool

# Extract msdata
u = msdata['uvw'][0]
v = msdata['uvw'][1]

# Take into account the conjugate symmetry of the UV plane
# Create PSF
uv_grid = np.zeros((grid_size_uv, grid_size_uv), dtype=np.complex128)

# Populate the UV grid
for u_val, v_val, w_val in zip(u, v, weights):
    u_index = int(u_val / uv_step) + grid_size_uv // 2
    v_index = int(v_val / uv_step) + grid_size_uv // 2
    if 0 <= u_index < grid_size_uv and 0 <= v_index < grid_size_uv:
        uv_grid[u_index, v_index] += w_val
    else:
        print(f'Warning: (u, v) = ({u_val + u_min}, {v_val + v_min}) is outside the UV grid')

# Compute the dirty beam using inverse FFT
dirty_beam = np.fft.ifftshift(np.fft.ifft2(np.fft.fftshift(uv_grid)))
dirty_beam = np.abs(dirty_beam)

