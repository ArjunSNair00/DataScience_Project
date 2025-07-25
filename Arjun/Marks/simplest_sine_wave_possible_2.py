import numpy as np
import matplotlib.pyplot as plt

x = np.arange(0, 1, 0.001)
y = np.sin(2*np.pi*x)

plt.plot(x, y)
plt.title("Sine Wave")
plt.xlabel("Time (s)")
plt.ylabel("Amplitude")
plt.grid(True)
plt.show()