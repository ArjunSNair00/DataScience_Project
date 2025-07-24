import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider

# Initial resolution
initial_points = 100

# Create figure and axes
fig, ax = plt.subplots()
plt.subplots_adjust(bottom=0.25)  # leave space for slider

# Generate initial data
x = np.linspace(0, 2 * np.pi, initial_points)
y = np.sin(x)

# Plot initial sine wave
line, = ax.plot(x, y)
ax.set_title("Sine Wave with Adjustable Resolution")
ax.set_xlabel("x (radians)")
ax.set_ylabel("sin(x)")
ax.grid(True)

# Slider axis: [left, bottom, width, height] in 0–1 figure coords
slider_ax = plt.axes([0.25, 0.1, 0.5, 0.03])
point_slider = Slider(
    ax=slider_ax,
    label='Number of Points',
    valmin=2,
    valmax=50,
    valinit=initial_points,
    valstep=1
)

# Update function for the slider
def update(val):
    points = int(point_slider.val)
    x = np.linspace(0, 2 * np.pi, points)
    y = np.sin(x)
    line.set_data(x, y)
    ax.relim()
    ax.autoscale_view()
    fig.canvas.draw_idle()

point_slider.on_changed(update)

plt.show()
