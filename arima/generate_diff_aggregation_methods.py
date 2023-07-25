import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Using ARIMA.py
from ARIMA import custom_ARIMA
def arima_custom(hospital, start_year = 2008, end_year = 2018, avg = 200, d =1, t = 0, mu = 0, sigma = 1, summary_type = "day"):
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

    if summary_type == "weekly":
        temp = res.head(int(len(res)*0.8)).resample('W', on='Date').sum().reset_index().sort_values(by='Date')
        # temp.drop(res.tail(1).index,inplace=True) # drop last row
        temp.drop(res.head(1).index,inplace=True) # drop first row
        # res["Date"] = pd.to_datetime(df.Date, format='%d/%m/%Y')

        temp = temp.set_index('Date').resample('D').ffill().reset_index()
        temp["Transfused"] = temp["Transfused"]/7
        res = pd.concat([temp, res.tail(int(len(res)*0.2))], ignore_index=True)

    elif summary_type == "biweekly":
        temp = res.head(int(len(res)*0.8)).resample('2W', on='Date').sum().reset_index().sort_values(by='Date')
        # res.drop(res.tail(1).index,inplace=True) # drop last row
        temp.drop(res.head(1).index,inplace=True) # drop first row
        
        temp = temp.set_index('Date').resample('D').ffill().reset_index()
        temp["Transfused"] = temp["Transfused"]/14
        res = pd.concat([temp, res.tail(int(len(res)*0.2))], ignore_index=True)

    elif summary_type == "monthly":
        temp = res.head(int(len(res)*0.8)).resample('M', on='Date').sum().reset_index().sort_values(by='Date')
        # res.drop(res.tail(1).index,inplace=True) # drop last row
        # res.drop(res.head(1).index,inplace=True) # drop first row

        temp = temp.set_index('Date').resample('D').ffill().reset_index()
        temp["Transfused"] = temp["Transfused"]/30.4
        res = pd.concat([temp, res.tail(int(len(res)*0.2))], ignore_index=True)

    elif summary_type == "quarterly":
        temp = res.head(int(len(res)*0.8)).resample('Q', on='Date').sum().reset_index().sort_values(by='Date')
        # res.drop(res.tail(1).index,inplace=True) # drop last row
        # res.drop(res.head(1).index,inplace=True) # drop first row        

        temp = temp.set_index('Date').resample('D').ffill().reset_index()
        temp["Transfused"] = temp["Transfused"]/91.25
        res = pd.concat([temp, res.tail(int(len(res)*0.2))], ignore_index=True)

    res["Location"] = hospital
    
    return res
        
def generate_with_arima():
    np.random.seed(4) # to get comparable results
    res = arima_custom("C1", sigma = 20)
    res.to_csv('./arima/C1.csv')
    plt.plot(res["Transfused"])
    plt.show()

    res = arima_custom("C2", sigma = 20, summary_type= "weekly")
    res.to_csv('./arima/C2.csv')
    plt.plot(res['Transfused'])
    plt.show()

    res = arima_custom("C3", sigma = 20, summary_type= "biweekly")
    res.to_csv("./arima/C4.csv")
    plt.plot(res['Transfused'])
    plt.show()

    res = arima_custom("C4", sigma = 20, summary_type= "monthly")
    res.to_csv("./arima/C4.csv")
    plt.plot(res['Transfused'])
    plt.show()

    res = arima_custom("C5", sigma = 20, summary_type="quarterly")
    res.to_csv("./arima/C5.csv")
    plt.plot(res['Transfused'])
    plt.show()

def main():
    generate_with_arima()    

if __name__ == "__main__":
    main()
