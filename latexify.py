import pandas as pd

def format_float(value):
    if isinstance(value, str):
        return value
    return f'{value:.2f}'

df = pd.read_csv('results.csv')
selected_columns = ["algorithm","target_size","final_size","time","r2_train","r2_test","rmse_train","rmse_test"]
df = df[selected_columns]
df.columns = ["Algorithm","Target Size","Final Size","Time","$R^2$(Train)","$R^2$(Test)","$RMSE$(Train)","$RMSE$(Test)"]
df = df.round(2)
df = df.map(format_float)
latex_table = df.to_latex(index=False)
print(latex_table)

