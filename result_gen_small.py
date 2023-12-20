import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import LinearRegression
from plot3 import plot_3d
from plot2 import plot_2d
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPRegressor

SOURCE = "data/dataset_66.csv"
DEST = "data/results_small_mlp.csv"


def get_score(X_train, y_train, X_test, y_test, i, j):
    model = get_model()
    new_X = X_train[:, [i, j]]
    model.fit(new_X, y_train)
    new_X = X_test[:, [i, j]]
    score = model.score(new_X, y_test)
    return score


def create_table(X,y):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=40)
    df = pd.DataFrame(columns=['i', 'j', 'score'])
    for i in range(X.shape[1]):
        for j in range(X.shape[1]):
            score = get_score(X_train, y_train, X_test, y_test, i, j)
            df.loc[len(df)] = [round(i),round(j),round(score,3)]
            print(f"Done {i},{j}")
        df.to_csv(DEST, index=False)
    return df


def gen_please():
    df = pd.read_csv(SOURCE)
    cols = [str(i) for i in range(66)]+["oc"]
    print(cols)
    df = df[cols].to_numpy()
    for i in range(df.shape[1]):
        scaler = MinMaxScaler()
        x_scaled = scaler.fit_transform(df[:, i].reshape(-1, 1))
        df[:, i] = np.squeeze(x_scaled)
    X = df[:,:-1]
    y = df[:,-1]
    create_table(X, y)


def get_model():
    #return LinearRegression()
    return MLPRegressor(hidden_layer_sizes=(10,8), random_state=41)


if __name__ == "__main__":
    gen_please()
    plot_3d(DEST)
    plot_2d(DEST)