import matplotlib.pyplot as plt
import pandas as pd
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm

source = "data/results_small_mlp.csv"
df = pd.read_csv(source)
i = df["i"].to_numpy()
j = df["j"].to_numpy()
mse = df["score"].to_numpy()
fig = plt.figure()
ax = fig.add_subplot(111, projection=Axes3D.name)

fig = plt.figure()

norm = plt.Normalize(min(mse), max(mse))
colors = cm.viridis(norm(mse))


#line = ax.plot_trisurf(i, j, mse, cmap='viridis', norm=norm, linewidth=0.2, alpha=0.4)
line = ax.scatter(i, j, mse, c=colors, alpha=0.2, label="3D Curve")

ax.set_xlabel("i")
ax.set_ylabel("j")
ax.set_zlabel("score")

mappable = cm.ScalarMappable(norm=norm, cmap=cm.viridis)
mappable.set_array(mse)
cbar = plt.colorbar(mappable, ax=ax, label="Score")

#ax.legend()

plt.show()
