import random
from collections import namedtuple

import torch
import torch.nn.functional as F
from torch import nn

Transition = namedtuple('Transition', ('state', 'action', 'next_state', 'reward'))


class ReplayMemory(object):

    def __init__(self, capacity):
        self.capacity = capacity
        self.memory = []
        self.position = 0

    def push(self, *args):
        if len(self.memory) < self.capacity:
            self.memory.append(None)
        self.memory[self.position] = Transition(*args)
        self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)


class DQN(nn.Module):

    def __init__(self, p=64):
        super(DQN, self).__init__()
        self.theta5 = torch.ones((2*p, 1), requires_grad=True)
        self.theta6 = torch.ones((p, p), requires_grad=True)
        self.theta7 = torch.ones((p, p), requires_grad=True)

    def forward(self, x: torch.Tensor, mu: torch.Tensor, w: torch.Tensor, adj_mat: torch.Tensor):
        t1 = mu.sum(dim=(0,)).repeat((x.shape[0], 1))
        t1 = t1.mm(self.theta6)  # (N, p)

        t2 = mu.mm(self.theta7)  # (N, p)

        t3 = torch.cat((t1, t2), dim=1)
        t3 = F.relu(t3)
        t3 = t3.mm(self.theta5)   # (N, 1)

        return t3
