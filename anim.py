import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
from matplotlib.animation import FuncAnimation
from PIL import Image
matplotlib.use("TkAgg")

res = pd.read_csv("fsdr.csv")

def get_bands(row):
    bands = []
    for i in range(1,6):
        bands.append(res.iloc[row][f"band_{i}"])
    return bands


bands = get_bands(0)
colour = ["red", "blue", "green", "black", "orange"]

df = pd.read_csv("data/dataset_min.csv")
signal = df.iloc[0].to_numpy()
start_index = list(df.columns).index("400")
signal = signal[start_index:]


fig = plt.figure(figsize=(8, 4))

line = plt.plot(signal)
points = []
for index, val in enumerate(bands):
    val = int(val)
    point = plt.scatter(val,signal[val],label=f"Band {index+1}", color=colour[index],s=100)
    points.append(point)

plt.legend()
plt.tight_layout()
title = plt.title('Epoch 0')

def update(frame):
    title.set_text(f'Epoch {frame+1}')
    bands = get_bands(frame+1)
    for index, val in enumerate(bands):
        val = int(val)
        points[index].set_offsets([[val, signal[val]]])
    return points,

#ani = FuncAnimation(fig, update, frames=len(res)-1, interval=100)
plt.subplots_adjust(top=0.85)
ani = FuncAnimation(fig, update, frames=300, interval=100, repeat=False)
#plt.show(block=False)
ani.save('animation.gif', writer='pillow', fps=10)
print("done")
