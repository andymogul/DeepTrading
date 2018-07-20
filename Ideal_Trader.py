import numpy as np

daily_price = np.genfromtxt('HyundaiCar.csv', delimiter=",")
initial_cash = 10000000
initial_date = 10

cash = initial_cash
stock_num = 0
portfolio_value = initial_cash
reward = 0
portfolio_queue = np.empty([1258])

date_pointer = 0
for i in range(initial_date, np.shape(daily_price)[0]-1):
    if daily_price[i+1] > daily_price[i]:
        cash -= daily_price[i]
        stock_num += 1
    elif daily_price[i+1] < daily_price[i]:
        cash += daily_price[i]
        stock_num -= 1
    new_portfolio_value = cash + daily_price[i+1] * stock_num
    reward += (new_portfolio_value - portfolio_value)
    portfolio_value = new_portfolio_value
    portfolio_queue[date_pointer] = portfolio_value
    
    date_pointer += 1

np.savetxt("Ideal_portfolio_output.csv", portfolio_queue, delimiter=",")

print(reward)

