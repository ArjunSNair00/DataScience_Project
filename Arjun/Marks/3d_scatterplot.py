import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits import mplot3d

#single points plotting
ax=plt.axes(projection="3d")

'''
#single point
ax.scatter(3,5,7)
'''

'''
#line plot
x=np.arange(0,50,0.1)
y=np.arange(0,50,0.1)
z=np.arange(0,50,0.1)
ax.scatter(x,y,z)
'''

# Define your own hex color palette
palette = [
  "#FF0000", "#000000", "#4C00FF"
]


points = 500


# Generate random points
x = np.random.randint(0, 100, (points,))
y = np.random.randint(0, 100, (points,))
z = np.random.randint(0, 100, (points,))

# Assign a random color from the palette to each point
colors = [palette[np.random.randint(0, len(palette))] for _ in range(points)]

ax.scatter(x, y, z, c=colors,marker='.',alpha=1,s=40)
#ax.plot(x, y, z)
plt.show()