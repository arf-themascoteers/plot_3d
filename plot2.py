import pandas as pd
import matplotlib.pyplot as plt

source = "data/results_mlp.csv"
df = pd.read_csv(source)
fig, ax = plt.subplots()
scatter = ax.scatter(df['i'], df['j'], c=df['mse'], cmap='viridis')

ax.set_xlabel('i')
ax.set_ylabel('j')
plt.colorbar(scatter, label='mse')
plt.show()