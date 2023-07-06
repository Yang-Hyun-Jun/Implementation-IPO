import torch
import argparse
import numpy as np 
import pandas as pd

from utils import make_batch
from utils import tensorize
from agent import IPOAgent
from memory import ReplayMemory
from env import Environment

parser = argparse.ArgumentParser()
parser.add_argument("--seed", type=int, default=1)
parser.add_argument("--num", type=int, default=90)
parser.add_argument("--lr1", type=float, default=1e-6)
parser.add_argument("--lr2", type=float, default=1e-6)
parser.add_argument("--tau", type=float, default=0.005)
parser.add_argument("--alpha", type=float, default=2.2)
parser.add_argument("--episode", type=float, default=2000)
parser.add_argument("--gamma", type=float, default=0.9)
parser.add_argument("--batch_size", type=float, default=128)
parser.add_argument("--memory_size", type=float, default=100000)
parser.add_argument("--balance", type=float, default=14560.05)
parser.add_argument("--holding", type=float, default=5)
args = parser.parse_args()

train_data = np.load(f'Data/train_data_tensor_{args.num}.npy')
test_data = np.load(f'Data/test_data_tensor_{args.num}.npy')

K = train_data.shape[1]
F = test_data.shape[2]

parameters= {
            "lr1":args.lr1, 
            "lr2":args.lr2, 
            "tau":args.tau, 
            "alpha":args.alpha,
            "gamma":args.gamma,
            "K":K, "F":F, 
            }

if __name__ == '__main__': 

    # Train Loop
    env = Environment(train_data)
    memory = ReplayMemory(args.memory_size)
    agent = IPOAgent(**parameters)

    PVs = []
    PFs = []

    for epi in range(1, args.episode+1):
        steps, cumr, cumc = 0, 0, 0
        a_loss, v_loss, c_loss = None, None, None
        state = env.reset(args.balance)

        while True:
            is_hold = steps % args.holding != 0

            if is_hold:
                action = np.zeros((K-1)) if is_hold else action
                next_state, _, _, done = env.step(action)
                state = next_state
                steps += 1

            else:
                action, sample, log_pi = agent.get_action(tensorize(state), env.portfolio)
                next_state, reward, cost, done = env.step(action, sample)
                env.initial_balance = env.portfolio_value
                transition = [state, sample, reward, cost, next_state, log_pi, done]
                memory.push(list(map(tensorize, transition)))

                cumr += reward[0]
                cumc += cost[0]
                state = next_state
                steps += 1

            if (len(memory)) >= args.batch_size:
                batch_data = make_batch(memory.sample(args.batch_size))
                v_loss, c_loss, a_loss = agent.update(*batch_data)
                agent.soft_target_update()

            if (epi == args.episode):
                PVs.append(env.portfolio_value)
                PFs.append(env.profitloss)

            if (epi == args.episode) & done[0]:
                pd.DataFrame({'Profitloss':PFs}).to_csv(f'Metrics/seed{args.seed}/Profitloss_Train')
                pd.DataFrame({'PV':PVs}).to_csv(f'Metrics/seed{args.seed}/Portfolio_Value_Train')
                torch.save(agent.net.score_net.state_dict(), f'Metrics/seed{args.seed}/actor.pth')

            if done[0]:
                print(f'epi:{epi}')
                print(f'a_loss:{a_loss}')
                print(f'v_loss:{v_loss}')
                print(f'c_loss:{c_loss}')
                print(f'cumr:{cumr}')
                print(f'cumc:{cumc} \n')
                break


    # Test Loop
    env = Environment(test_data)
    agent.net.eval()
    
    steps = 0
    PVs = []
    PFs = []
    POs = []
    Cs = []

    state = env.reset(args.balance)
    
    while True:
        is_hold = steps % args.holding != 0

        if is_hold:
            action = np.zeros((K-1))
            next_state, _, _, done = env.step(action)
            state = next_state
            steps += 1

        else:
            action, sample, _ = agent.get_action(tensorize(state), env.portfolio, 'mode')
            next_state, reward, cost, done = env.step(action, sample)
            env.initial_balance = env.portfolio_value
            state = next_state
            steps += 1

        PVs.append(env.portfolio_value)
        PFs.append(env.profitloss)
        POs.append(env.portfolio)
        Cs.append(cost)

        if done[0]:
            pd.DataFrame({'Profitloss':PFs}).to_csv(f'Metrics/seed{args.seed}/Profitloss_Test_IPO')
            pd.DataFrame({'PV':PVs}).to_csv(f'Metrics/seed{args.seed}/Portfolio_Value_Test_IPO')
            pd.DataFrame({'C':Cs}).to_csv(f'Metrics/seed{args.seed}/Cost_Test_IPO')
            pd.DataFrame(POs).to_csv(f'Metrics/seed{args.seed}/Portfolios_Test_IPO')
            break
            

        
        


        


