import random
from collections import namedtuple

import matplotlib.pyplot as plt
import networkx as nx


Env_transition = namedtuple('Environment_transition', ('s_t', 'a', 's_tp1', 'r', 'cum_r_t', 'is_terminated'))


class Environment:

    def __init__(self, name: str, n_min=5, n_max=20, p_edge=0.15):
        self.name = name
        self.n_nodes = random.randint(n_min, n_max)
        self.graph = nx.erdos_renyi_graph(self.n_nodes, p_edge)
        self.s = []
        self.v_bar = list(range(self.n_nodes))

    def reset(self, *args, **kwargs):
        self.__init__(*args, **kwargs)

    def make_transition(self, node):
        cum_r_t = self.get_cumulated_reward()
        old_state = self.s.copy()
        self.v_bar.remove(node)
        self.s.append(node)
        transition = Env_transition(s_t=old_state, a=node, s_tp1=self.s.copy(), r=-1, cum_r_t= cum_r_t,
                                    is_terminated=self.is_terminated())
        return transition

    def get_cumulated_reward(self):
        pass

    def is_terminated(self):
        pass

    def show(self):
        nx.draw(self.graph, with_labels=True, font_weight='bold')
        plt.show()


class MVCEnvironment(Environment):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.cover_graph = nx.Graph()
        self.cover_graph.add_nodes_from(self.graph.nodes)

    def make_transition(self, node):
        transition = super().make_transition(node)
        u = transition.a
        self.cover_graph.add_node(u)
        self.cover_graph.add_edges_from([(u, v) for v in self.graph.neighbors(u)])
        return transition

    def is_terminated(self):
        return set(self.cover_graph.edges) == set(self.graph.edges)

    def get_cumulated_reward(self):
        return -len(self.s)

    def show(self):
        fig, ax = plt.subplots(1, 2, figsize=(8, 4))
        nx.draw(self.graph, with_labels=True, font_weight='bold', ax=ax[0])
        nx.draw(self.cover_graph, with_labels=True, font_weight='bold', node_color='g', ax=ax[1])
        plt.show()
