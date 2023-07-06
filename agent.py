import torch 
import torch.nn as nn

from network import Network

class IPOAgent:
    """
    Interior Point policy Optimization (IPO)
    """
    def __init__(self, **kwargs):
        
        self.K = kwargs["K"]
        self.F = kwargs["F"]
        self.lr1 = kwargs["lr1"]
        self.lr2 = kwargs["lr2"]
        self.tau = kwargs["tau"]
        self.alpha = kwargs["alpha"]
        self.gamma = kwargs["gamma"] 
        
        self.con_t = 50.0
        
        self.net = Network(self.K, self.F).to('cuda')
        self.target_net = Network(self.K, self.F).to('cuda')
        self.target_net.load_state_dict(self.net.state_dict())

        self.huber = nn.SmoothL1Loss()
        self.optim = torch.optim.Adam([{'params':self.net.score_net.parameters(), 'lr':self.lr1},
                                       {'params':self.net.value_net.parameters(), 'lr':self.lr2},
                                       {'params':self.net.const_net.parameters(), 'lr':self.lr2}])
        
    
    def get_action(self, s, p, mode=None):
        with torch.no_grad():
            sample = self.net.sampling(s, mode).squeeze(0)
            log_pi = self.net.log_prob(s, sample)
            sample = sample.cpu().numpy()
            log_pi = log_pi.cpu().numpy()
            action = (sample - p)[1:]
        return action, sample, log_pi
    
    
    def update(self, s, p, r, c, ns, log_pi, done):
        """
        IPO Actor, Critic Update
        PPO off-policy style
        """
        eps_clip = 0.2

        log_pi_ = self.net.log_prob(s, p).unsqueeze(1)
        ratio = torch.exp(log_pi - log_pi_)

        # Two Critic Loss
        with torch.no_grad():
            next_v = self.target_net.value(ns)
            v_target = r + self.gamma * next_v * (1-done)
            v_target = v_target * torch.clamp(ratio.detach(), 1-eps_clip, 1+eps_clip)

            next_c = self.target_net.c_value(ns)
            c_target = c + self.gamma * next_c * (1-done)
            c_target = c_target * torch.clamp(ratio.detach(), 1-eps_clip, 1+eps_clip)

        value = self.net.value(s)
        c_value = self.net.c_value(s)

        v_loss = self.huber(value, v_target)
        c_loss = self.huber(c_value, c_target)

        # Actor Loss
        Jc = c_value.mean()
        
        if Jc - self.alpha <= 0:
            td_advantage = c + self.gamma * self.net.c_value(ns) * (1-done) - c_value
            surr1 = ratio * td_advantage
            surr2 = torch.clamp(ratio, 1-eps_clip, 1+eps_clip) * td_advantage
            a_loss = torch.min(surr1, surr2)
            a_loss = torch.mean(a_loss)
            
        else:
            td_advantage = r + self.gamma * self.net.value(ns) * (1-done) - value 
            barrier = torch.log(Jc - self.alpha) / self.con_t
            surr1 = ratio * td_advantage
            surr2 = torch.clamp(ratio, 1-eps_clip, 1+eps_clip) * td_advantage
            a_loss = -torch.min(surr1, surr2)-barrier
            a_loss = torch.mean(a_loss)

        self.optim.zero_grad()
        loss_all = v_loss + c_loss + a_loss
        loss_all.backward()
        self.optim.step()
        return v_loss.item(), c_loss.item(), a_loss.item()

    def soft_target_update(self):
        for param, target_param in zip(self.net.parameters(), self.target_net.parameters()):
            target_param.data.copy_(self.tau*param.data + (1-self.tau)*target_param.data)








