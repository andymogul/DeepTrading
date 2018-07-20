import numpy as np

target_dates = 100

first_daily_price = np.genfromtxt("Samsung_fin.csv", delimiter = ",")
second_daily_price = np.genfromtxt("Samsung_fin.csv", delimiter = ",")

var_first = np.empty([np.shape(first_daily_price)[0] - target_dates])
var_second = np.empty([np.shape(second_daily_price)[0] - target_dates])
cov = np.empty([np.shape(first_daily_price)[0] - target_dates])
hedge_ratio = np.empty([np.shape(first_daily_price)[0] - target_dates])


for i in range(0, np.shape(first_daily_price)[0] - target_dates - 1) :
    var_first[i] = np.var(first_daily_price[i:i+target_dates])
    var_second[i] = np.var(second_daily_price[i:i+target_dates])
    cov[i] = np.cov(first_daily_price[i:i+target_dates], second_daily_price[i:i+target_dates])[0][1]
    hedge_ratio[i] = (cov[i] / var_second[i]) * first_daily_price[i+target_dates] / second_daily_price[i+target_dates]

np.savetxt("aaaaa.csv", hedge_ratio, delimiter=",")