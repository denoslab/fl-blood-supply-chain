import math
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import random

# Using ARIMA.py
from ARIMA import custom_ARIMA
def arima_custom(hospital,  phi, theta, start_year = 2008, end_year = 2018, avg = 200, d =0, t = 0, mu = 0, sigma = 1):
    res = pd.DataFrame()

    for i in range(start_year, end_year):
        start = str(i) + "-01-01"
        end = str(i) + "-12-31"
        df = pd.DataFrame({"Date": pd.date_range(start, end)})
        res = pd.concat([res, df], ignore_index=True)
    
    n = len(res.index)
    # phi = np.array([0.4087, -0.5934, -0.3317, -0.2706, -0.2912]) # AR part using 5 lag
    # theta = np.array([-1.2007, 0.9153]) # MA part using 2 lag
    # np.random.seed(42) # to get comparable results

    transfused = custom_ARIMA(phi = phi, theta = theta, d = d, t = t, mu = mu, sigma= sigma, n = n) # simulate time series
    transfused = [i[0] + avg for i in transfused]
    min_transfused = min(transfused)
    if min_transfused < 0:
        transfused = -min_transfused + transfused + 1 # add 1 at end because MAPE will blow up if there are zeros

    res["Transfused"] = transfused
    res["Location"] = hospital
    
    return res
        

def generate_with_arima():
    np.random.seed(42) # to get comparable results
    phi = np.array([1.2459,-0.9967])
    theta = np.array([-2.0716 ,2.0332, -0.8939, 0.0826, -0.0475])
    res = arima_custom("Cluster_1_Hospital_1", phi=phi, theta=theta,d=1, sigma = math.sqrt(2.7472))
    res.to_csv('./arima/C1.csv')
    plt.plot(res["Transfused"])
    plt.show()

    res = arima_custom("Cluster_1_Hospital_2", phi=phi, theta=theta,d=1, sigma = math.sqrt(2.7472))
    res.to_csv('./arima/C2.csv')
    plt.plot(res['Transfused'])
    plt.show()

    phi = np.array([0.2145, 0.0671, 0.0705, 0.0478])
    theta = np.array([-0.9862])
    res = arima_custom("Cluster_2_Hospital_1", phi=phi, theta=theta,d=1, sigma = math.sqrt(1.4037))
    res.to_csv("./arima/C3.csv")
    plt.plot(res['Transfused'])
    plt.show()

    res = arima_custom("Cluster_2_Hospital_2", phi=phi, theta=theta,d=1, sigma = math.sqrt(1.4037))
    res.to_csv("./arima/C4.csv")
    plt.plot(res['Transfused'])
    plt.show()

    res = arima_custom("Cluster_2_Hospital_3", phi=phi, theta=theta,d=1, sigma = math.sqrt(1.4037))
    res.to_csv("./arima/C5.csv")
    plt.plot(res['Transfused'])
    plt.show()

def main():
    generate_with_arima()    

if __name__ == "__main__":
    main()
