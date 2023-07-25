import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from statsmodels.tsa.arima.model import ARIMA
import random

# Using ARIMA.py
from ARIMA import custom_ARIMA
def arima_custom(hospital, start_year = 2008, end_year = 2018, avg = 200, d =1, t = 0, mu = 0, sigma = 1, zeros = "normal", percent_zeros = 10):
    res = pd.DataFrame()

    for i in range(start_year, end_year):
        start = str(i) + "-01-01"
        end = str(i) + "-12-31"
        df = pd.DataFrame({"Date": pd.date_range(start, end)})
        res = pd.concat([res, df], ignore_index=True)
    
    n = len(res.index)
    phi = np.array([0.4087, -0.5934, -0.3317, -0.2706, -0.2912]) # AR part using 5 lag
    theta = np.array([-1.2007, 0.9153]) # MA part using 2 lag
    # np.random.seed(42) # to get comparable results

    transfused = custom_ARIMA(phi = phi, theta = theta, d = d, t = t, mu = mu, sigma= sigma, n = n) # simulate time series
    transfused = [i[0] + avg for i in transfused]
    min_transfused = min(transfused)
    if min_transfused < 0:
        transfused = -min_transfused + transfused + 1 # add 1 at end because MAPE will blow up if there are zeros

    if zeros == "random":
        count = 0
        for i in range(round(len(transfused)*0.8)):
            if random.randint(1,100) <= percent_zeros:
                transfused[i] = 0
                count+=1
        print(count, len(transfused))

    res["Transfused"] = transfused
    res["Location"] = hospital

    if zeros == "fri-sun":
        res["day_of_week"] = res["Date"].dt.day_name()
        
        res.loc[res["day_of_week"] == "Friday", "Transfused"] = 0 
        res.loc[res["day_of_week"] == "Saturday", "Transfused"] = 0 
        res.loc[res["day_of_week"] == "Sunday", "Transfused"] = 0 
        # print(res["date"])
        # print(res["day_of_week"])
        res.drop(columns = ['day_of_week'], inplace=True)
    return res

def generate_with_arima():
    np.random.seed(4) # to get comparable results
    res = arima_custom("C1", sigma = 20, zeros = "random", percent_zeros= 10)
    res.to_csv('./arima/C1.csv')
    plt.plot(res["Transfused"])
    plt.show()

    res = arima_custom("C2", sigma = 20 )
    res.to_csv('./arima/C2.csv')
    plt.plot(res['Transfused'])
    plt.show()

    res = arima_custom("C3", sigma = 20, zeros="random", percent_zeros= 20)
    res.to_csv("./arima/C3.csv")
    plt.plot(res['Transfused'])
    plt.show()

    res = arima_custom("C4", sigma = 20 )
    res.to_csv("./arima/C4.csv")
    plt.plot(res['Transfused'])
    plt.show()

    res = arima_custom("C5", sigma = 20, zeros = "random", percent_zeros= 30)
    res.to_csv("./arima/C5.csv")
    plt.plot(res['Transfused'])
    plt.show()

def main():
    generate_with_arima()    

if __name__ == "__main__":
    main()
