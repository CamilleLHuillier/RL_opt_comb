import os
from collections import namedtuple

from agents import N_step_transition, DQNAgent, RandomAgent
from environments import MVCEnvironment
from graph_utils import get_adjacency_matrix, ADJACENCY_MATRIX, WEIGHTS_MATRIX
from rl_utils import Test

Step = namedtuple('Step', ['s', 'a', 'cum_r'])

PATH = '/Users/camille/dev/supelec/RL/HW2/com_opt_dqn/'

# Training
L = 5*10**4

# Graph properties
N_MIN_NODES = 10
N_MAX_NODES = 15

# DQN agent
MEMORY_CAPACITY = 2000
EPSILON = 1.
EPSILON_DECAY = 0.95
EMBEDDING_SIZE = 64
N_STEP = 5
GAMMA = 1.
LEARNING_RATE = 0.0001
BATCH_SIZE = 32

# S2V
N_ITER_S2V = 3

envs = {}

agent = DQNAgent(mem_capacity=MEMORY_CAPACITY, epsilon=EPSILON, emb_size=EMBEDDING_SIZE, n_iter_s2v=N_ITER_S2V,
                 gamma=GAMMA, learning_rate=LEARNING_RATE)

random_agent = RandomAgent()

test = Test(n_graphs=80, n_nodes_max=100)

n_max_nodes = N_MAX_NODES

agent_scores = []
random_agent_scores = []

for e in range(1, L):
    # set a new environment
    env_name = f'ep_{e}'
    env = MVCEnvironment(name=env_name, n_min=N_MIN_NODES, n_max=n_max_nodes)
    adj_mat = get_adjacency_matrix(env.graph)
    weights = adj_mat
    envs[env_name] = {ADJACENCY_MATRIX: adj_mat, WEIGHTS_MATRIX: weights}

    trajectory = []

    t = 0
    loss = 0

    while not env.is_terminated():

        best_action = agent.pick_action(s=env.s, v_bar=env.v_bar, w=weights, adj_mat=adj_mat, train=True)
        env_transition = env.make_transition(best_action)
        step = Step(s=env_transition.s_t, a=env_transition.a, cum_r=env_transition.cum_r_t)
        trajectory.append(step)

        if t >= N_STEP:
            if env.is_terminated():
                s_ter = env_transition.s_tp1
                cum_r_ter = env.get_cumulated_reward()
                for k in range(N_STEP):
                    # add N_STEP transitions to terminal states
                    step_tmn = trajectory[-(k+1)]
                    n_step_transition = N_step_transition(s_tmn=step_tmn.s, v_tmn=step_tmn.a, cum_r_tmn=step_tmn.cum_r,
                                                          s_t=s_ter, cum_r_t=cum_r_ter, is_terminated=True,
                                                          env_name=env.name)
                    agent.memorize(transition=n_step_transition)

            else:
                # add one transitions to regular states
                step_tmn = trajectory[-N_STEP]
                step_t = trajectory[-1]
                n_step_transition = N_step_transition(s_tmn=step_tmn.s, v_tmn=step_tmn.a, cum_r_tmn=step_tmn.cum_r,
                                                      s_t=step_t.s, cum_r_t=step_t.cum_r, is_terminated=False,
                                                      env_name=env.name)
                agent.memorize(transition=n_step_transition)

            loss, y_med, q_v_tmn_med = agent.reinforce(batch_size=BATCH_SIZE, envs=envs)

        t += 1

    if e % 100 == 0:
        agent_score = test.evaluate_policy(agent)
        random_agent_score = test.evaluate_policy(random_agent)

        agent_scores.append(agent_score)
        random_agent_scores.append(random_agent_score)

        print(f'episode {e}, n_nodes: {env.n_nodes}, iter {t}, loss {loss}, epsilon {agent.epsilon}, '
              f'agent_score {agent_score}, random_agent_score {random_agent_score}')

        agent.save(folder_path=os.path.join(PATH, 'models'), episode=e)

        agent.epsilon *= EPSILON_DECAY
        if e < 3000:
            n_max_nodes += 2
        agent.epsilon *= EPSILON_DECAY

    if e > 2000:  # avoid keeping to many graphs in memory
        del envs[f'ep_{e-200}']
