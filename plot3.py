import matplotlib.pyplot as plt
import pandas as pd
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm


def plot_3d(source):
    df = pd.read_csv(source)
    df = df[df['i'] < df['j']]
    i = df["i"].to_numpy() + 1
    j = df["j"].to_numpy() + 1
    mse = df["score"].to_numpy()
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection=Axes3D.name)
    FS = 12
    fig = plt.figure()

    norm = plt.Normalize(min(mse), max(mse))
    colors = cm.viridis(norm(mse))


    #line = ax.plot_trisurf(i, j, mse, cmap='viridis', norm=norm, linewidth=0.2, alpha=0.4)
    line = ax.scatter(i, j, mse, c=colors, alpha=0.2)

    ax.set_xlabel("Band Index 1", fontsize=FS)
    ax.set_ylabel("Band Index 2", fontsize=FS)
    ax.set_zlabel(r"$R^2$", fontsize=FS)

    ax.set_xticks([1,11,21,31,41,51,61])
    ax.set_yticks([1,11,21,31,41,51,61])

    ax.tick_params(axis='x', labelsize=FS)
    ax.tick_params(axis='y', labelsize=FS)
    ax.tick_params(axis='z', labelsize=FS)

    #ax.grid(False)

    mappable = cm.ScalarMappable(norm=norm, cmap=cm.viridis)
    mappable.set_array(mse)
    cbar = plt.colorbar(mappable, ax=ax, label="Score",orientation='horizontal', fraction=0.04, pad=0.1)
    cbar.set_label(r"$R^2$", fontsize=FS)
    cbar.ax.tick_params(labelsize=FS)

    highlight_x = 5
    highlight_y = 15
    highlight_z = 0.7
    scatter = ax.scatter([highlight_x], [highlight_y], [highlight_z], color='green', s=20, label='Best performance', zorder = 10000)
    #scatter.set_zorder(10)
    #ax.legend()
    text = 'Best performance'
    ax.text(highlight_x-20, highlight_y+1, highlight_z, text, color='green', fontsize=FS, ha='left', va='bottom', zorder=10)
    plt.subplots_adjust(bottom=0.3)
    plt.show()


if __name__ == "__main__":
    plot_3d("data/results_small_mlp.csv")