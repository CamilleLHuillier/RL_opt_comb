import random
from collections import namedtuple
from typing import List

import numpy as np
import torch
import torch.optim as optim

from graph_utils import WEIGHTS_MATRIX, ADJACENCY_MATRIX
from rl_utils import DQN, ReplayMemory, Struc2Vec

N_step_transition = namedtuple('N_step_transition', ['s_tmn', 'v_tmn', 'cum_r_tmn', 's_t', 'cum_r_t', 'is_terminated',
                                                     'env_name'])


class DQNAgent:

    def __init__(self, mem_capacity: int = 200, epsilon: float = 0.1, emb_size: int = 64, n_iter_s2v: int = 5,
                 gamma: float = 0.99, learning_rate: float = 0.001):
        self.emb_size = emb_size
        self.dqn = DQN(emb_dim=self.emb_size)
        self.memory = ReplayMemory(capacity=mem_capacity)
        self.epsilon = epsilon
        self.n_iter_s2v = n_iter_s2v
        self.gamma = gamma
        self.learning_rate = learning_rate
        self.struc2vec = Struc2Vec(p=self.emb_size)
        self.optimizer = optim.SGD(params=[self.struc2vec.theta1, self.struc2vec.theta2, self.struc2vec.theta3,
                                           self.struc2vec.theta4, self.dqn.theta5, self.dqn.theta6, self.dqn.theta7],
                                   lr=self.learning_rate)

    def get_mu(self, s: List[int], w: torch.Tensor, adj_mat: torch.Tensor):
        x = self.struc2vec.s2x(s=s, n_nodes=w.shape[0])
        mu = torch.zeros((w.shape[0], self.emb_size))
        for i in range(self.n_iter_s2v):
            mu = self.struc2vec(x=x, mu=mu, w=w, adj_mat=adj_mat)
        return mu

    def pick_action(self, s: List[int], v_bar: List[int], w: torch.Tensor, adj_mat: torch.Tensor):
        """Follow the epsilon greedy strategy"""
        if random.random() < self.epsilon:
            return random.choice(v_bar)
        else:
            mu = self.get_mu(s=s, w=w, adj_mat= adj_mat)
            q_hat = self.dqn(mu).detach().numpy().reshape(-1)
            i_max = np.argmax(q_hat[v_bar])
            return v_bar[i_max]

    def memorize(self, transition: N_step_transition):
        self.memory.push(transition)

    def reinforce(self, batch_size: int, envs: dict):
        samples = self.memory.sample(batch_size=batch_size)
        loss = 0

        for i in range(len(samples)):
            w = envs[samples[i].env_name][WEIGHTS_MATRIX]
            adj_mat = envs[samples[i].env_name][ADJACENCY_MATRIX]

            mu_tmn = self.get_mu(s=samples[i].s_tmn, w=w, adj_mat=adj_mat)
            mu_t = self.get_mu(s=samples[i].s_t, w=w, adj_mat=adj_mat)

            q_hat_tmn = self.dqn(mu_tmn)
            q_hat_t = self.dqn(mu_t)

            # TODO: cumulated reward, include r_tmn ? include r_t ?
            q_hat_t_max = q_hat_t.detach().numpy().max()
            y = samples[i].cum_r_t - samples[i].cum_r_tmn + self.gamma*q_hat_t_max

            v_tmn = samples[i].v_tmn
            q_v_tmn = q_hat_tmn[v_tmn, 0]

            loss += (y - q_v_tmn)**2

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return loss
