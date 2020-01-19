# Import required packages
import numpy as np
import pandas as pd
from gym.utils import seeding
import gym
from gym import spaces

# Read and process the raw data set
data = pd.read_csv('D:/jiashi/adj_close.csv')
stock_num = len(data.columns)-2

# Replace NA stock price with 0
newdata = data.fillna(0)

# Load the file for print the balance, reward and logging
output = open("D:/jiashi/balance.txt", "w+")
reward_file = open("D:/jiashi/reward.txt", "w+")
logging = open("D:/jiashi/log.txt", "w+")
shares_file = open("D:/jiashi/shares.txt", "w+")

# 252 trading days a year for 7 years of training, which include 2 years of validation.

training_start = 0
training_end = 252 * 7

validation_start = 252 * 5
validation_end = 252 * 7
validation_length = validation_end-validation_start

# 2 years of test
test_start = 252 * 7
test_end = 252 * 9

train_data = newdata.iloc[training_start:training_end,:]

validation_data = newdata.iloc[validation_start:validation_end,:]

test_data = newdata.iloc[test_start:test_end,:]

# Environment parameters
iteration = 0
TRADE_THRESHOLD = 5 # buy or sell maximum 5 shares per time
INITIAL_BALANCE = 1000000
TRADING_FEE = 0.0002 # the rate of trading fee per transaction
INITIAL_TRADE = 7 # decrease the reward to day number / INITIAL_TRADE for first INITIAL_TRADE days

class StockValEnv(gym.Env):
    metadata = {'render.modes': ['human']}

    def __init__(self, day = 0, money = INITIAL_BALANCE , scope = 1):
        self.day = day
        
        self.action_space = spaces.Box(low = np.array([-TRADE_THRESHOLD]*stock_num), 
                                       high = np.array([TRADE_THRESHOLD]*stock_num),
                                       dtype=np.int8) 

        # observation matrix:
        # obs_space[0:training_end,0] : total asset of day i before transaction
        # obs_space[1:training_end,1:506]1-505: price of SP 500 stocks every day
        # obs_space[1:training_end,506]: price of SP 500 index as the baseline
        # obs_space[0,1:506]: number of shares per stock at day i
        self.observation_space = spaces.Box(low=0, high=np.inf, 
                                            shape = (validation_length+1,stock_num+2),
                                            dtype=np.float16)
        
        # Create the initial state
        temp = np.zeros([validation_length+1,stock_num+2])
        temp[self.day+1,1:stock_num+2] = validation_data.iloc[self.day, 1:stock_num+2]
        
        self.terminal = False
        
        self.state = temp
        self.reward = 0
        
        self.balance_memory = [INITIAL_BALANCE]

        self.reset()
        self._seed()


    def _transaction_stock(self, action):
        balance = self.balance_memory[self.day]
        
        
        for i in range(action.shape[0]):
            share = int(action[i])
            # Sell the ith stock
            if share < 0:
                if self.state[0, i+1] > 0:
                    # Calculate shares to sell
                    sell = min(abs(share), self.state[0, i+1])
                    
                    # Update balance
                    balance += self.state[self.day+1, i+1] * int(sell) * (1 - TRADING_FEE)
                    
                    # Update holdings
                    self.state[0, i+1] -= int(sell)
                    
                else:
                    pass
                        
            # Buy the ith stock
            elif share > 0:
                # Check the price of the stock
                if (self.state[self.day+1, i+1]==0):
                    pass
                else:
                    # Calculate shares to buy
                    available_amount = balance // (self.state[self.day+1, i + 1] * (1 + TRADING_FEE))
                    buy = int(min(share, available_amount))
                    
                    # Update balance
                    balance -= self.state[self.day+1, i + 1]*buy*(1+TRADING_FEE)
                        
                    # Update holdings
                    self.state[0, i+1] += buy
                    
            # Hold
            else:
                pass
        self.balance_memory.append(balance)
        
    def step(self, action):
        # print(self.day)
        self.terminal = self.day >= validation_length - 1
        # print(actions)

        if self.terminal:
            print(self.state[:,0], file = output)
            return self.state, self.reward, self.terminal,{}

        else:
            
            # Do the transaction
            
            self._transaction_stock(action)
            
            # Update price and total asset after the transaction
            self.day += 1
            self.state[self.day+1, 1:stock_num+2] = validation_data.iloc[self.day, 1:stock_num+2] 
            self.state[self.day+1, 0] = self.balance_memory[self.day] + np.sum(self.state[0, 1:stock_num+1]*self.state[self.day+1, 1:stock_num+1])
            
            # Calculate the annualized reward
            sd = np.std((self.state[0:self.day+1, 0]-INITIAL_BALANCE)/INITIAL_BALANCE)
            earn = (self.state[self.day+1, 0] - INITIAL_BALANCE)/ INITIAL_BALANCE
            baseline = (self.state[self.day+1, stock_num+1] - self.state[1, stock_num+1])/self.state[1, stock_num+1]
            
            if sd == 0:
                sharp = 0
            else:
                sharp = (earn - baseline) / sd * np.sqrt(255/self.day)
                
            if self.day <= INITIAL_TRADE:
                self.reward = sharp * self.day / INITIAL_TRADE
            else:
                self.reward = sharp
            
            print("{0}, {1}".format(self.day, self.reward), file = logging)
        return self.state, self.reward, self.terminal, {}

    def reset(self):

        
        self.terminal = False
        
        self.reward = 0
        
        self.balance_memory = [INITIAL_BALANCE]
        
        self.day = 0
        # Create the initial state
        temp = np.zeros([validation_length+1,stock_num+2])
        temp[self.day+1,1:stock_num+2] = validation_data.iloc[self.day, 1:stock_num+2]
        
        self.state = temp
        
        return self.state
    
    def render(self, mode='human'):
        
        return self.state

    def _seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]
    