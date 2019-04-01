import random

import torch

import networkx as nx
import numpy as np

ADJACENCY_MATRIX = 'adjacency_matrix'
WEIGHTS_MATRIX = 'weights_matrix'


def get_adjacency_matrix(graph: nx.Graph):
    n_nodes = graph.number_of_nodes()
    adj_matrix = np.zeros((n_nodes, n_nodes))
    for u, neighbours in graph.adjacency():
        for v in neighbours.keys():
            adj_matrix[u, v] = 1
            adj_matrix[v, u] = 1
    return torch.Tensor(adj_matrix)


def draw_graph(n_nodes: int = 10):
    if random.random() < 0.33:
        return nx.erdos_renyi_graph(n=n_nodes, p=0.15)
    elif random.random() < 0.66:
        return nx.powerlaw_cluster_graph(n=n_nodes, m=4, p=0.05)
    else:
        return nx.barabasi_albert_graph(n=n_nodes, m=4)
