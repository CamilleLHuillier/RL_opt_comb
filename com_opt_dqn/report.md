<h2> Reinforcement learning project </h2>

<h3>Struc2Vec</h3>

1. Advantages:
    - Generic training procedure for graph combinatory optimization
    - Ability to deal with instances with various sizes, nature (scale-free graph, ER graph)
    - Ability to automatically learn state of the art heuristics

2. Consequences:
    - S is continuous
    - states' representations have to be learnt

<br> 
<h3>Training procedure- fitted Q learning</h3>

1. Model free approach:
    - from the agent perspective the set of states is continuous and infinite S = R^{2p} 
    - the state of actions is finite A = V
    - the MDP (p(s' | s, a), r(s, a)) is unknown to the agent
    
2. Fitted Q learning:
    - look up tables can't be used (S is continuous). Instead we use a parameterized regressor (neural network) and a cost function to estimate Q(s, a). At this point, common gradient descent techniques (like the ’backpropagation’ learning rule) can be applied to adjust the weights of a neural network in order to minimize the error.
    - <b>convergence guarantees</b> ?
    - on-line updates usually doesn't perform well. Indeed the neural network computes global representations of the (state, action) pairs. A weight change induced by an update in a certain part of the state space might influence the values in arbitrary other regions and therefore destroy the effort done so far in other regions



References:
- http://ml.informatik.uni-freiburg.de/former/_media/publications/rieecml05.pdf


<h3>Implementation</h3>

1. mu_t computation (iterations? n_iter? convergence?)
2. n-step Q-learning (justification, interpretation)
3. theta_i and mu_0 initialization (Glorot initialization?) ?
4. embedding dimension ?


<h3>Bonus</h3>

1. epsilon-greedy ? (epsilon decay)
2. learning-rate