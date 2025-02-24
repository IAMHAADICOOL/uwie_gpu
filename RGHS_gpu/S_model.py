import cupy as cp
import matplotlib.pyplot as plt

# Generate x values using CuPy
x = cp.arange(-128, 127)

# Compute y values using CuPy vectorized operations
y = x * cp.power(2, (1 - cp.abs(x / 128)))

# Convert to NumPy for plotting
x = cp.asnumpy(x)
y = cp.asnumpy(y)

# Plot the curve
plt.figure(figsize=(8, 6))
plt.axis([-128, 127, -128, 127])
plt.title('S-model Curve Function', fontsize=20)
plt.xlabel('Input Value', fontsize=20)
plt.ylabel('Output Value', fontsize=20)
plt.plot(x, y, color='red')
plt.show()
