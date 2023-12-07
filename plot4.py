import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.widgets import Slider
import pandas as pd
import matplotlib
matplotlib.use("TkAgg")

df = pd.read_csv('data/results_small_mlp.csv')

fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')
scatter = ax.scatter(df['i'], df['j'], df['score'],  c=df['score'], cmap='viridis', marker='o', alpha=0.25)

ax.set_xlabel('i')
ax.set_ylabel('j')
ax.set_zlabel('Score')

axcolor = 'lightgoldenrodyellow'
ax_i = plt.axes([0.1, 0.01, 0.65, 0.03], facecolor=axcolor)
ax_j = plt.axes([0.1, 0.06, 0.65, 0.03], facecolor=axcolor)

s_i = Slider(ax_i, 'i', 0, 66, valinit=df['i'].iloc[0], valstep=1)
s_j = Slider(ax_j, 'j', 0, 66, valinit=df['j'].iloc[0], valstep=1)

marked = None

def update(val):
    global marked
    i_val = s_i.val
    j_val = s_j.val
    scatter._offsets3d = (df['i'], df['j'], df['score'])
    if marked is not None:
        marked.remove()
    marked = ax.scatter(i_val, j_val, df.loc[(df['i'] == i_val) & (df['j'] == j_val), 'score'], c='r', marker='o', s=100)
    fig.canvas.draw_idle()

s_i.on_changed(update)
s_j.on_changed(update)

plt.show()
