import numpy as np




class Single_StockSpace:
    
    def __init__(self):
        self.daily_price = np.genfromtxt('.csv', delimiter=",")

        self.state = None
        self.observation = None

        self.ob_range = 10
        self.buy_cost = 0.00015
        self.sell_cost = 0.003

        self.initial_cash = 0
        self.initial_stock = 0
        
        # Action Space : [1,0,0] = Buy   [0,1,0] = Sell   [0,0,1] = Hold
        self.action_space = np.zeros(3)


    # state : 전날 10일의 가격로 해보자.
    def step(self, action):
        assert action in {0,1,2}, 'action %d is not in action_space' % action

        
        state = self.state
        date, cash, stock_num, asset_value = state
        observation = self.daily_price[date - self.ob_range : date]
        
        if action == 0:
            cash -= (self.daily_price[date]*(1+self.buy_cost))
            stock_num += 1
        elif action == 1:
            cash += (self.daily_price[date]*(1-self.sell_cost))
            stock_num -= 1
        
        if date < self.end_date-1:
            new_asset_value = cash + stock_num * self.daily_price[date+1]
        else:
            new_asset_value = asset_value
        
        date += 1
        self.state = (date, cash, stock_num, new_asset_value)

        reward = new_asset_value - asset_value
 
        done = date >= self.end_date
        done = bool(done)

        return observation, np.array(self.state), reward, done

    def reset(self, chk):
        if chk == 0:
            self.initial_date = 10
            self.end_date = 1000
        if chk == 1:
            self.initial_date = 1000
            self.end_date = np.shape(self.daily_price)[0]


        self.state = (self.initial_date, self.initial_cash, self.initial_stock, self.initial_cash)
        self.observation = self.daily_price[self.initial_date - self.ob_range : self.initial_date]
        return np.array(self.state), self.observation




#거래비용 0.3%

class Pair_StockSpace:
    

    def __init__(self):
        self.first_daily_price = np.genfromtxt('Samsung_fin.csv', delimiter=",")
        self.second_daily_price = np.genfromtxt('NH_fin.csv', delimiter=",")
        self.first_normalized_price = np.genfromtxt('Samsung_fin_normalized.csv', delimiter=",")
        self.second_normalized_price = np.genfromtxt('NH_fin_normalized.csv', delimiter=",")

        self.hedge_ratio = np.genfromtxt('Hedge_ratio_samsung2nh.csv', delimiter=",")
        self.state = None
        self.observation = None

        self.ob_range = 5
        self.buy_cost = 0.00015
        self.sell_cost = 0.003

        self.initial_cash = 0
        self.initial_first_stock = 0
        self.initial_second_stock = 0
        # Action Space : [1,0,0] = Buy&Sell   [0,1,0] = Sell&Buy   [0,0,1] = Hold
        self.action_space = np.zeros(3)

    # state : 전날 10일의 가격로 해보자.
    
    def step(self, action):
        assert action in {0,1,2}, 'action %d is not in action_space' % action
        
        state = self.state
        date, cash, first_stock_num, second_stock_num, asset_value = state
        
        observation = self.first_normalized_price[date - self.ob_range : date] - self.second_normalized_price[date - self.ob_range : date]
        
        if action == 0:
            cash -= (self.first_daily_price[date]*(1+self.buy_cost))
            cash += ((self.second_daily_price[date]*self.hedge_ratio[date])*(1-self.sell_cost))
            first_stock_num += 1
            second_stock_num -= self.hedge_ratio[date]
            
        elif action == 1:
            cash += (self.first_daily_price[date]*(1-self.sell_cost))
            cash -= ((self.second_daily_price[date]*self.hedge_ratio[date])*(1+self.buy_cost))
            first_stock_num -= 1
            second_stock_num += self.hedge_ratio[date]
        
        if date < self.end_date-1:
            new_asset_value = cash + first_stock_num * self.first_daily_price[date+1] + second_stock_num * self.second_daily_price[date+1]
        else:
            new_asset_value = asset_value
        
        date += 1
        self.state = (date, cash, first_stock_num, second_stock_num, new_asset_value)

        reward = new_asset_value - asset_value

        done = date >= self.end_date
        done = bool(done)

        return observation, np.array(self.state), reward, done

    def reset(self, chk):
        if chk == 0:
            self.initial_date = 10
            self.end_date = 1000
        if chk == 1:
            self.initial_date = 1000
            self.end_date = np.shape(self.first_daily_price)[0]
        
        self.state = (self.initial_date, self.initial_cash, self.initial_first_stock, self.initial_second_stock, self.initial_cash)
        self.observation = self.first_normalized_price[self.initial_date - self.ob_range : self.initial_date] - self.second_normalized_price[self.initial_date - self.ob_range : self.initial_date]
        return np.array(self.state), self.observation







        


