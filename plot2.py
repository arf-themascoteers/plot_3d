import pandas as pd
import matplotlib.pyplot as plt

source = "data/results_small_lr.csv"
df = pd.read_csv(source)
fig, ax = plt.subplots()
scatter = ax.scatter(df['i'], df['j'], c=df['score'], cmap='viridis')

ax.set_xlabel('i')
ax.set_ylabel('j')
plt.colorbar(scatter, label='score')
plt.show()