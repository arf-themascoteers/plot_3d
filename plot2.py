import pandas as pd
import matplotlib.pyplot as plt


def plot_2d(source):
    df = pd.read_csv(source)
    fig, ax = plt.subplots()
    scatter = ax.scatter(df['i'], df['j'], c=df['score'], cmap='viridis')

    ax.set_xlabel('i')
    ax.set_ylabel('j')
    plt.colorbar(scatter, label='score')
    plt.show()