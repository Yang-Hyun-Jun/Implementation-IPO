import torch
import numpy as np

def tensorize(array):
    tensor = torch.tensor(array[np.newaxis]).float().to("cuda")
    return tensor 

def make_batch(transition):
    x = list(zip(*transition))
    x = list(map(torch.cat, x))
    return x

