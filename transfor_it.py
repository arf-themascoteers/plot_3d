import pandas as pd

SMALL = "data/dataset_66_min.csv"
BIG = "data/dataset.csv"


def transform_small_please():
    df = pd.read_csv(SMALL)
    cols = list(df.columns)
    cols = cols[1:]+ cols[0:1]
    df = df[cols]
    df.to_csv(SMALL, index=False)


def transform_big_please():
    wvs = []
    spec = 400
    while spec <= 2499.5:
        n_spec = spec
        if int(n_spec) == spec:
            n_spec = int(n_spec)
        wavelength = str(n_spec)
        wvs.append(wavelength)
        spec = spec + 0.5

    df = pd.read_csv(BIG)
    cols = list(df.columns)
    cols = cols[1:] + cols[0:1]
    df = df[cols]
    df.columns = [str(i) for i in range(4200)] + ["oc"]
    df.to_csv(BIG, index=False)


if __name__ == "__main__":
    transform_big_please()