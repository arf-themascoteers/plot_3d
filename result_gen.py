import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler


SMALL = "data/dataset_66_min.csv"
BIG = "data/dataset.csv"
SOURCE = SMALL


def gen_please():
    data = pd.read_csv(SMALL).to_numpy()
    for i in range(data.shape[1]):
        scaler = MinMaxScaler()
        x_scaled = scaler.fit_transform(data[:, i].reshape(-1, 1))
        data[:, i] = np.squeeze(x_scaled)
    X = data[:,:-1]
    y = data[:,-1]



if __name__ == "__main__":
    gen_please()