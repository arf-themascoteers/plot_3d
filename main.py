source = "data/dataset_66_min.csv"

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np

x1 = np.linspace(0, 10, 100)
x2 = np.linspace(0, 10, 100)
y = np.sin(x1) + np.cos(x2)

fig = plt.figure()
ax = fig.add_subplot(111, projection=Axes3D.name)

ax.plot(x1, x2, y, label='3D Curve')

ax.set_xlabel('X1')
ax.set_ylabel('X2')
ax.set_zlabel('Y')

ax.legend()

plt.show()
