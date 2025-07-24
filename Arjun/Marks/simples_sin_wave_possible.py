import matplotlib.pyplot as plt
import numpy as np

f = 10                    # Frequency in Hz
w = 2 * np.pi * f
x = np.arange(0, 1, 0.001)  # 1 second total duration
y = np.sin(w * x)

plt.plot(x, y)
plt.title(f"{f} Hz Sine Wave (1 second)")
plt.xlabel("Time (s)")
plt.ylabel("Amplitude")
plt.grid(True)
plt.show()