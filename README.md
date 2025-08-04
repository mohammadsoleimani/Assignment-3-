OverviewGridPathRL is a reinforcement learning project implementing SARSA and Q-learning algorithms to learn an optimal policy for navigating a 5x5 grid world. The agent starts at (4,0), aims to reach terminal states at (0,0) or (0,4) (reward 0) while avoiding red states at (2,0), (2,1), (2,3), (2,4) (reward -20, reset to start), with normal steps or wall collisions yielding -1 reward. The optimal path is 8 steps through (2,2), yielding a total reward of -8.EnvironmentGrid: 5x5, 25 states.
Start State: (4,0).
Terminal States: (0,0), (0,4) (reward 0).
Red States: (2,0), (2,1), (2,3), (2,4) (reward -20, reset to (4,0)).
Actions: Up, down, left, right.
Rewards:Normal step or wall collision: -1.
Red state: -20.
Step to terminal state: -1.

Goal: Reach a terminal state via the optimal 8-step path (e.g., (4,0) → (3,0) → (3,1) → (3,2) → (2,2) → (1,2) → (0,2) → (0,1) → (0,0), reward -8).

AlgorithmsSARSA: On-policy algorithm, updates Q-values based on the next action taken.
Q-learning: Off-policy algorithm, updates Q-values based on the best next action.
Parameters:Learning rate (α\alpha\alpha
): 0.1.
Discount factor (γ\gamma\gamma
): 0.99.
Epsilon decay: ϵt=max⁡(0.01,0.1/(1+t/1000))\epsilon_t = \max(0.01, 0.1 / (1 + t/1000))\epsilon_t = \max(0.01, 0.1 / (1 + t/1000))
, where (t) is the episode number.
Episodes: 1000.
Random seed: 42 for reproducibility.

ImplementationLanguage: Python 3.
Dependencies: numpy, matplotlib, seaborn.
Files:gridpathrl.py: Main script with environment, agents, training, and visualization.

Key Features:Epsilon-greedy action selection with linear decay for exploration.
Corrected reward function to ensure -8 reward for the 8-step path.
Visualization of learning curves, trajectories, policies, and reward distributions.

PlotsLearning Curves Comparison: Smoothed rewards (50-episode window) for SARSA and Q-learning.
Recent Performance: Raw rewards for the last 200 episodes.
SARSA Trajectory: Path from (4,0) to (0,0) using SARSA’s greedy policy.
Q-Learning Trajectory: Path from (4,0) to (0,0) using Q-learning’s greedy policy.
SARSA Policy: Best actions per state as arrows on the grid.
Q-Learning Policy: Best actions per state, typically identical to SARSA.
Reward Distribution: Histogram of rewards (last 500 episodes).
Convergence Analysis: Reward standard deviation (100-episode window).

Expected OutputTrajectories: Identical for both algorithms: ([(4,0), (3,0), (3,1), (3,2), (2,2), (1,2), (0,2), (0,1), (0,0)]), length 9 (8 steps).
Test Performance (100 episodes):

SARSA test performance: -8.00 ± 0.00
Q-Learning test performance: -8.00 ± 0.00

Reward Behavior:Q-learning: Converges to -8.0 ± ~0.2–0.3 by ~400–500 episodes, smoother.
SARSA: Converges to -8.0 ± ~0.4–0.5 by ~500–600 episodes, more variable.

Policy Similarity: ~100% identical actions due to deterministic grid and seed 42.



Output:Console: Training progress, trajectory details, performance metrics, and policy comparison.
Plots: Eight visualizations saved/displayed (learning curves, trajectories, policies, reward distributions).

NotesThe deterministic grid and fixed seed ensure identical trajectories. In stochastic environments, trajectories may differ slightly.
Epsilon decay slightly accelerates convergence and reduces variance but is not critical for this small grid.
Q-learning converges faster with less variance than SARSA due to its off-policy nature.

RequirementsPython 3.6+
Libraries: numpy, matplotlib, seaborn

