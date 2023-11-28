import matplotlib.pyplot as plt
import pandas as pd
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm

source = "data/results.csv"
df = pd.read_csv(source)
i = df["i"].tolist()
j = df["j"].tolist()
mse = df["mse"].tolist()
fig = plt.figure()
ax = fig.add_subplot(111, projection=Axes3D.name)

fig = plt.figure()

norm = plt.Normalize(min(mse), max(mse))
colors = cm.viridis(norm(mse))


line = ax.plot_trisurf(i, j, mse, cmap='viridis', norm=norm, linewidth=0.2, alpha=0.4)


ax.set_xlabel("i")
ax.set_ylabel("j")
ax.set_zlabel("mse")

mappable = cm.ScalarMappable(norm=norm, cmap=cm.viridis)
mappable.set_array(mse)
cbar = plt.colorbar(mappable, ax=ax, label="MSE")

#ax.legend()

plt.show()
