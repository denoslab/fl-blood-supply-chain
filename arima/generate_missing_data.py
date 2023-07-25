import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.stattools import acf, pacf
from sklearn.model_selection import train_test_split
import random

# Using ARIMA.py
from ARIMA import custom_ARIMA
def arima_custom(hospital, start_year = 2008, end_year = 2018, avg = 200, d =1, t = 0, mu = 0, sigma = 1, missing_data= False , percent_missing = 10):
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


    res["Transfused"] = transfused
    res["Location"] = hospital

    if missing_data:
        count = 0
        for (index, row) in res[:round(len(res)*0.8)].iterrows():
            if random.randint(1,100) <= percent_missing:
                res.drop(index, inplace=True)
                count +=1
        print("Missing ", count, " values", " Dataframe length", len(res))

    return res

def generate_with_arima():
    np.random.seed(42) # to get comparable results
    res = arima_custom("Foothills Hospital", sigma = 20, missing_data= True, percent_missing=20)
    res.to_csv('./arima/FoothillsHospital.csv')
    plt.plot(res["Transfused"])
    plt.show()

    res = arima_custom("Rockyview Hospital", sigma = 20)
    res.to_csv('./arima/RockyviewHospital.csv')
    plt.plot(res['Transfused'])
    plt.show()

    res = arima_custom("Alberta Children Hospital", sigma = 20, missing_data= True, percent_missing=30)
    res.to_csv("./arima/AlbertaChildrenHospital.csv")
    plt.plot(res['Transfused'])
    plt.show()

    res = arima_custom("University Hospital", sigma = 20 )
    res.to_csv("./arima/UniversityHospital.csv")
    plt.plot(res['Transfused'])
    plt.show()

    res = arima_custom("Bowness Hospital", sigma = 20, missing_data= True, percent_missing= 40)
    res.to_csv("./arima/BownessHospital.csv")
    plt.plot(res['Transfused'])
    plt.show()

def main():
    generate_with_arima()    

if __name__ == "__main__":
    main()
