import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits import mplot3d

ax=plt.axes(projection="3d")

x=np.arange(0,50,0.1)
y=np.arange(0,50,0.1)
z=np.sin(x)*np.sin(y)

ax.plot(x, y, z)
plt.show()