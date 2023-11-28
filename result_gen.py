import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import LinearRegression
from sklearn.neural_network import MLPRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error


SMALL = "data/dataset_66_min.csv"
BIG = "data/dataset.csv"
SOURCE = SMALL


def create_table(X,y):
    train_x, test_x, train_y, test_y = train_test_split(X,y, test_size=0.1, random_state=2)
    df = pd.DataFrame(columns=['i', 'j', 'mse'])
    for i in range(X.shape[1]):
        for j in range(X.shape[1]):
            train_x_new = train_x[:,[i,j]]
            test_x_new = test_x[:,[i,j]]
            model = get_model()
            model.fit(train_x_new, train_y)
            y_pred = model.predict(test_x_new)
            mse = mean_squared_error(test_y, y_pred)
            df.loc[len(df)] = [round(i),round(j),round(mse,3)]
            print(f"Done {i},{j}")
    return df

def gen_please():
    data = pd.read_csv(SMALL).to_numpy()
    for i in range(data.shape[1]):
        scaler = MinMaxScaler()
        x_scaled = scaler.fit_transform(data[:, i].reshape(-1, 1))
        data[:, i] = np.squeeze(x_scaled)
    X = data[:,:-1]
    y = data[:,-1]
    df = create_table(X, y)
    df.to_csv("data/results.csv", index=False)



def get_model():
    #return LinearRegression()
    return MLPRegressor(hidden_layer_sizes=(10,), random_state=41)

if __name__ == "__main__":
    gen_please()