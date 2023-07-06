import torch
import torch.nn as nn
import numpy as np

from torch.distributions.dirichlet import Dirichlet

class Network(nn.Module):
    def __init__(self, K, F):
        super().__init__()

        self.K = K
        self.F = F

        self.score_net = nn.Sequential(
            nn.Linear(F * K, 512),
            nn.LeakyReLU(),
            nn.Linear(512, 256),
            nn.LeakyReLU(),
            nn.Linear(256, K),
            )

        self.value_net = nn.Sequential(
            nn.Linear(F * K, 512),
            nn.LeakyReLU(),
            nn.Linear(512, 256),
            nn.LeakyReLU(),
            nn.Linear(256, 1),            
            )

        self.const_net = nn.Sequential(
            nn.Linear(F * K, 512),
            nn.LeakyReLU(),
            nn.Linear(512, 256),
            nn.LeakyReLU(),
            nn.Linear(256, 1),
            )

    def entropy(self, state):
        """
        Dirichlet Dist의 현재 entropy
        """
        alpha = self.alpha(state)
        dirichlet = Dirichlet(alpha)
        return dirichlet.entropy()
    
    def c_value(self, state):
        """
        Expected Sum of Cost
        """
        state = state.reshape(-1, self.F * self.K)
        c = self.const_net(state)
        return c

    def value(self, state):
        """
        Critic의 Value
        """
        state = state.reshape(-1, self.F * self.K)
        v = self.value_net(state)
        return v

    def alpha(self, state):
        """
        Dirichlet Dist의 Concentration Parameter
        """
        state = state.reshape(-1, self.F * self.K)
        scores = self.score_net(state).reshape(-1, self.K)
        scores = torch.clamp(scores, -40., 500.)
        alpha = torch.exp(scores) + 1.0
        return alpha

    def log_prob(self, state, portfolio):
        """
        Dirichlet Dist에서 샘플의 log_prob
        """
        alpha = self.alpha(state)
        dirichlet = Dirichlet(alpha)
        return dirichlet.log_prob(portfolio)

    def sampling(self, state, mode=None):
        """
        Dirichlet Dist에서 포트폴리오 샘플링
        """
        alpha = self.alpha(state).detach()
        dirichlet = Dirichlet(alpha)
    
        if mode == "mean":
            sampled_p = dirichlet.mean

        elif mode == "mode":
            sampled_p = dirichlet.mode

        else:
            sampled_p = dirichlet.sample([1])[0]
        
        return sampled_p
    

if __name__ == '__main__':
    F = 9
    K = 11

    s = torch.rand(size=(1, F, K))

    net = Network(K, F)
    sample = net.sampling(s)
    log_pi = net.log_prob(s, sample)

    print(log_pi.shape)
    print(sample.shape)
    