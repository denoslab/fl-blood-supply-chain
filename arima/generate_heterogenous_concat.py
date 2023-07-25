import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import random

# Using ARIMA.py
from ARIMA import custom_ARIMA
def arima_custom(hospital, phi1, theta1, phi2, theta2, start_year = 2008, end_year = 2018, avg = 200, d =1, t = 0, mu = 0, sigma = 1):
    res = pd.DataFrame()

    for i in range(start_year, end_year):
        start = str(i) + "-01-01"
        end = str(i) + "-12-31"
        df = pd.DataFrame({"Date": pd.date_range(start, end)})
        res = pd.concat([res, df], ignore_index=True)
    
    n = len(res.index)//2
    if len(res.index) % 2 != 0:
        n1, n2= n, n+1
    # np.random.seed(42) # to get comparable results

    transfused = custom_ARIMA(phi = phi1, theta = theta1, d = d, t = t, mu = mu, sigma= sigma, n = n1) # simulate time series
    res_transfused = [i[0] + avg for i in transfused]

    transfused = custom_ARIMA(phi = phi2, theta = theta2, d = d, t = t, mu = mu, sigma= sigma, n = n2) # simulate time series
    res_transfused = res_transfused + [i[0] + avg for i in transfused]

    min_transfused = min(res_transfused)
    if min_transfused < 0:
        res_transfused = -min_transfused + res_transfused + 1 # add 1 at end because MAPE will blow up if there are zeros

    res["Transfused"] = res_transfused
    res["Location"] = hospital

    return res

def generate_with_arima():
    np.random.seed(42) # to get comparable results
    phi1 = np.array([0.2145, 0.0671, 0.0705, 0.0478]) # AR part using 5 lag
    theta1 = np.array([-0.9862]) # MA part using 2 lag

    phi2 = np.array([1.2459  , -0.9967 ]) # AR part using 2 lag
    theta2 = np.array([-2.0716 , 2.0332, -0.8939, 0.0826, -0.0475]) # MA part using 5 lag

    phi3 = np.array([0.1036,-0.0437 ])
    theta3 = np.array([ -0.8937])

    res = arima_custom("Foothills Hospital", phi1, theta1, phi2, theta2, sigma = 20)
    res.to_csv('./arima/FoothillsHospital.csv')
    plt.plot(res["Transfused"])
    plt.show()

    res = arima_custom("Rockyview Hospital",phi2, theta2, phi3, theta3,  sigma = 20)
    res.to_csv('./arima/RockyviewHospital.csv')
    plt.plot(res['Transfused'])
    plt.show()

    res = arima_custom("Alberta Children Hospital", phi1, theta1, phi3, theta3, sigma = 20)
    res.to_csv("./arima/AlbertaChildrenHospital.csv")
    plt.plot(res['Transfused'])
    plt.show()

    res = arima_custom("University Hospital", phi1, theta1, phi2, theta2, sigma = 20 )
    res.to_csv("./arima/UniversityHospital.csv")
    plt.plot(res['Transfused'])
    plt.show()

    res = arima_custom("Bowness Hospital", phi2, theta2, phi3, theta3, sigma = 20)
    res.to_csv("./arima/BownessHospital.csv")
    plt.plot(res['Transfused'])
    plt.show()

def main():
    generate_with_arima()    

if __name__ == "__main__":
    main()
