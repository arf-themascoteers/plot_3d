import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import LinearRegression
from sklearn.neural_network import MLPRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score


SMALL = "data/dataset_66_min.csv"
BIG = "data/dataset.csv"
SOURCE = BIG


def create_table(X,y):
    df = pd.DataFrame(columns=['i', 'j', 'score'])
    for i in range(0,X.shape[1], 10):
        for j in range(0,X.shape[1],10):
            model = get_model()
            new_X = X[:,[i,j]]
            model.fit(new_X,y)
            score = model.score(new_X,y)
            df.loc[len(df)] = [round(i),round(j),round(score,3)]
            print(f"Done {i},{j}")
    return df


def gen_please():
    data = pd.read_csv(SOURCE).to_numpy()
    for i in range(data.shape[1]):
        scaler = MinMaxScaler()
        x_scaled = scaler.fit_transform(data[:, i].reshape(-1, 1))
        data[:, i] = np.squeeze(x_scaled)
    X = data[:,:-1]
    y = data[:,-1]
    df = create_table(X, y)
    df.to_csv("data/results_mlp_big.csv", index=False)


def get_model():
    #return LinearRegression()
    return MLPRegressor(hidden_layer_sizes=(10,), random_state=41)


if __name__ == "__main__":
    gen_please()