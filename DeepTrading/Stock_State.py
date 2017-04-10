import numpy as np

class StockSpace:
    
    def __init__(self):
        self.daily_price = np.genfromtxt('HyundaiCar.csv', delimiter=",")
        self.state = None
        self.initial_cash = 100
        # Action Space : [1,0,0] = Buy   [0,1,0] = Sell   [0,0,1] = Hold
        self.action_space = np.zeros(3)
        
        




    def _step(self, action):
        assert action in {'0', '1', '2'}, 'action %d is not in action_space' % action
        
        state = self.state
        date, cash, stock_num, asset_value = state
        if action == 0:
            cash -= self.daily_price[date]
            stock_num += 1
        elif action == 1:
            cash += self.daily_price[date]
            stock_num -= 1
        new_asset_value = cash + self.daily_price[date+1]
        
        date += 1
        self.state = (date, cash, stock_num, new_asset_value)

        reward = new_asset_value - asset_value

        return np.array(self.state), reward

    def _reset(self):
        self.state = (0, self.initial_cash, 0, self.initial_cash)
        return np.array(self.state)

    







        


