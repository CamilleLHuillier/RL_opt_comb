import random
from typing import List

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

from environments import MVCEnvironment
from graph_utils import draw_graph, get_adjacency_matrix


class Struc2Vec(nn.Module):

    def __init__(self, p: int = 64):
        super(Struc2Vec, self).__init__()
        self.theta1 = torch.Tensor(np.random.randn(*(1, p)) * 0.01)
        self.theta1.requires_grad = True
        self.theta2 = torch.Tensor(np.random.randn(*(p, p)) * 0.01)
        self.theta2.requires_grad = True
        self.theta3 = torch.Tensor(np.random.randn(*(p, p)) * 0.01)
        self.theta3.requires_grad = True
        self.theta4 = torch.Tensor(np.random.randn(*(1, 1, p)) * 0.01)
        self.theta4.requires_grad = True

    def forward(self, x: torch.Tensor, mu: torch.Tensor, w: torch.Tensor, adj_mat: torch.Tensor):
        t1 = self.theta1.mul(x)  # (N, p)

        t2 = adj_mat.mm(mu)
        t2 = t2.mm(self.theta2)  # (N, p)

        t3 = self.theta4.mul(w.unsqueeze(2))
        t3 = F.relu(t3)
        t3 = t3.sum(dim=1)
        t3 = t3.mm(self.theta3)  # (N, p)
        return F.relu(torch.add(t1, t2.add(t3)))

    @staticmethod
    def s2x(s: List[int], n_nodes: int):
        """One hot encode the state [1, 4, 5] to the Tensor x"""
        eye = np.eye(n_nodes)
        x = eye[s, :].sum(axis=0).reshape((-1, 1))
        return torch.Tensor(x)


class ReplayMemory(object):

    def __init__(self, capacity):
        self.capacity = capacity
        self.memory = []
        self.position = 0

    def push(self, transition):
        if len(self.memory) < self.capacity:
            self.memory.append(None)
        self.memory[self.position] = transition
        self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size):
        if batch_size >= len(self.memory):
            return random.sample(self.memory, len(self.memory))
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)


class DQN(nn.Module):

    def __init__(self, emb_dim=64):
        super().__init__()
        self.theta5 = torch.Tensor(np.random.randn(*(2 * emb_dim, 1))*0.01)
        self.theta5.requires_grad = True
        self.theta6 = torch.Tensor(np.random.randn(*(emb_dim, emb_dim)) * 0.01)
        self.theta6.requires_grad = True
        self.theta7 = torch.Tensor(np.random.randn(*(emb_dim, emb_dim)) * 0.01)
        self.theta7.requires_grad = True

    def forward(self, mu: torch.Tensor):
        t1 = mu.sum(dim=(0,)).repeat((mu.shape[0], 1))
        t1 = t1.mm(self.theta6)  # (N, emb_dim)

        t2 = mu.mm(self.theta7)  # (N, emb_dim)

        t3 = torch.cat((t1, t2), dim=1)
        t3 = F.relu(t3)
        t3 = t3.mm(self.theta5)  # (N, 1)
        return t3


class Test:

    def __init__(self, n_graphs: int = 100, n_nodes_max: int = 200):
        np.random.seed(1234)

        self.graphs = []

        for i in range(n_graphs):
            n_nodes = random.randint(10, n_nodes_max)
            graph = draw_graph(n_nodes=n_nodes)
            self.graphs.append(graph)

    def evaluate_policy(self, agent):
        cumulated_reward = 0

        for graph in self.graphs:
            env = MVCEnvironment(name='test', graph=graph)
            adj_mat = get_adjacency_matrix(graph=graph)
            weights = adj_mat

            while not env.is_terminated():
                best_action = agent.pick_action(s=env.s, v_bar=env.v_bar, w=weights, adj_mat=adj_mat, train=False)
                _ = env.make_transition(best_action)

            cumulated_reward += env.get_cumulated_reward()

        return cumulated_reward
