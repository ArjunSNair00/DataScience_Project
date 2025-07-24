import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.animation import FuncAnimation
from matplotlib.widgets import Slider, CheckButtons

# Create figure and 3D axis
fig = plt.figure(figsize=(10, 7))
ax = fig.add_subplot(111, projection='3d')
plt.subplots_adjust(left=0.1, bottom=0.25)

# Number of points
points = 500

# Initial positions
x = np.random.uniform(0, 100, points)
y = np.random.uniform(0, 100, points)
z = np.random.uniform(0, 100, points)

# Color palette
palette = ["#FF5733", "#EEFF00", "#4C00FF"]
colors = [palette[np.random.randint(0, len(palette))] for _ in range(points)]

# Initial scatter plot
sc = ax.scatter(x, y, z, c=colors, marker='.')

# Axis limits
ax.set_xlim(0, 100)
ax.set_ylim(0, 100)
ax.set_zlim(0, 100)

# Movement settings
movement_enabled = [True]
movement_distance = [1.0]  # float value

# Update function
def update(frame):
    if not movement_enabled[0]:
        return sc,

    dx = np.random.uniform(-movement_distance[0], movement_distance[0], points)
    dy = np.random.uniform(-movement_distance[0], movement_distance[0], points)
    dz = np.random.uniform(-movement_distance[0], movement_distance[0], points)

    global x, y, z
    x = np.clip(x + dx, 0, 100)
    y = np.clip(y + dy, 0, 100)
    z = np.clip(z + dz, 0, 100)

    sc._offsets3d = (x, y, z)
    return sc,

# Animation at fixed 60 FPS (~16ms)
ani = FuncAnimation(fig, update, interval=16, blit=False)

# Slider for float movement distance
ax_dist = plt.axes([0.25, 0.15, 0.5, 0.03])
slider_dist = Slider(ax_dist, 'Movement Distance', 0.0, 10.0, valinit=1.0, valstep=0.1)

def update_distance(val):
    movement_distance[0] = slider_dist.val

slider_dist.on_changed(update_distance)

# Checkbox: toggle movement
ax_check = plt.axes([0.85, 0.15, 0.1, 0.1])
check = CheckButtons(ax_check, ['Moving'], [True])

def toggle_movement(label):
    movement_enabled[0] = not movement_enabled[0]

check.on_clicked(toggle_movement)

plt.show()
