from collections import namedtuple

from agents import N_step_transition, DQNAgent
from environments import MVCEnvironment
from graph_utils import get_adjacency_matrix, ADJACENCY_MATRIX, WEIGHTS_MATRIX

Step = namedtuple('Step', ['s', 'a', 'cum_r'])

# Training
L = 1

# Erdos-Renyi graph properties
N_MIN_NODES = 20
N_MAX_NODES = 50
P_EDGE = 0.15

# DQN agent
MEMORY_CAPACITY = 10
EPSILON = 0.1
EMBEDDING_SIZE = 10
N_STEP = 3
GAMMA = 1.  # TODO: gamma ?
LEARNING_RATE = 0.001
BATCH_SIZE = 5

# S2V
N_ITER_S2V = 1

envs = {}

agent = DQNAgent(mem_capacity=2, epsilon=EPSILON, emb_size=EMBEDDING_SIZE, n_iter_s2v=N_ITER_S2V, gamma=GAMMA,
                 learning_rate=LEARNING_RATE)

for e in range(L):

    # set a new environment
    env_name = f'ep_{e}'
    env = MVCEnvironment(name=env_name)
    adj_mat = get_adjacency_matrix(env.graph)
    weights = adj_mat
    envs[env_name] = {ADJACENCY_MATRIX: adj_mat, WEIGHTS_MATRIX: weights}

    trajectory = []

    t = 0

    while not env.is_terminated():

        best_action = agent.pick_action(s=env.s, v_bar=env.v_bar, w=weights, adj_mat=adj_mat)
        env_transition = env.make_transition(best_action)
        step = Step(s=env_transition.s_t, a=env_transition.a, cum_r=env_transition.cum_r_t)
        trajectory.append(step)

        if t >= N_STEP:
            step_tmn = trajectory[-N_STEP]
            step_t = trajectory[-1]
            n_step_transition = N_step_transition(s_tmn=step_tmn.s, v_tmn=step_tmn.a, cum_r_tmn=step_tmn.cum_r,
                                                  s_t=step_t.s, cum_r_t=step_t.cum_r,
                                                  is_terminated=env_transition.is_terminated, env_name=env.name)
            agent.memorize(transition=n_step_transition)
            loss = agent.reinforce(batch_size=BATCH_SIZE, envs=envs)

        t += 1
