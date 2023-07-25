import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.stattools import acf, pacf


def data_vis(filepath: str):
        
    # Loading in data
    res = pd.read_csv(filepath, index_col = [1])
    res.index = pd.to_datetime(res.index)

    transfused_list = res["Transfused"].values.tolist()
    plt.plot(transfused_list)
    plt.show()

if __name__ == "__main__":

    data_vis('./arima/C1.csv')
    data_vis('./arima/C2.csv')
    data_vis('./arima/C3.csv')
    data_vis('./arima/C4.csv')
    data_vis('./arima/C5.csv')