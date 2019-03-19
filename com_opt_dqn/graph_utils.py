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
