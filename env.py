import numpy as np
import pmenv

class Environment(pmenv.Environment):
    def __init__(self, stock_tensor=None):
        """
        L: Len of data
        K: Num of portfolio assets
        F: Num of features

        stock_tensor: tensor with shape (L, K, F)
        observation: feature matrix with shape (K, F) 
        """
        super().__init__(stock_tensor)

    def reset(self, balance):
        observation = super().reset(balance)
        state = self.get_state(observation, self.portfolio)
        return state

    def step(self, action, sample=None):
        next_observation, reward, done = super().step(action)
        next_state = self.get_state(next_observation, self.portfolio)
        cost = None if sample is None else np.array([self.get_cost(sample)])
        return next_state, reward, cost, done

    def get_cost(self, p):
        """
        Constrained Optimization을 위한 Cost 함수
        """
        cost = 1 if p[0] <= 0.003 else 0 
        return cost
    
    def get_state(self, observation, portfolio):
        """
        Price, Portfolio, Cushion으로 State 인코딩 
        """
        portfolio = portfolio[:,np.newaxis]
        state = np.concatenate([observation, portfolio], axis=1)
        return state 