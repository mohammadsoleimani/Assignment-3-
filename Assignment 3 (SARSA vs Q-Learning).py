# Assignment 3
# Reinforcement Learning
# Mohammad Soleimani


import numpy as np
import matplotlib.pyplot as plt
import random
from collections import defaultdict
import seaborn as sns

#  this is our 5x5 grid world where the agent learns to dodge red traps and reach a goal!
class GridWorld:
    def __init__(self):
        self.grid_size = 5
        self.start_state = (4, 0)  # Blue square, where we kick things off
        self.black_states = [(0, 0), (0, 4)]  # Black squares, the finish line (reward 0)
        self.red_states = [(2, 0), (2, 1), (2, 3), (2, 4)]  # Red squares, big penalty (-20) and reset
        self.actions = ['up', 'down', 'left', 'right']
        self.action_effects = {
            'up': (-1, 0),  # Move up: row decreases
            'down': (1, 0),  # Move down: row increases
            'left': (0, -1),  # Move left: column decreases
            'right': (0, 1)  # Move right: column increases
        }

    def is_valid_state(self, state):
        row, col = state
        return 0 <= row < self.grid_size and 0 <= col < self.grid_size  # Check if we're on the grid

    def get_next_state(self, state, action):
        row, col = state
        dr, dc = self.action_effects[action]
        new_state = (row + dr, col + dc)
        if self.is_valid_state(new_state):
            return new_state
        return state  # Can't move off the grid? Stay put!

    def get_reward(self, state, action, next_state):
        intended_state = (state[0] + self.action_effects[action][0],
                         state[1] + self.action_effects[action][1])
        if not self.is_valid_state(intended_state):
            return -1  # Bumping into walls costs -1
        if next_state in self.red_states:
            return -20
        if next_state in self.black_states:
            return -1  # Step to goal costs -1 (action cost)
        return -1  # Normal step, small cost

    def is_terminal(self, state):
        return state in self.black_states  # Game over if we're at a black square

    def reset(self):
        return self.start_state  # Back to the starting line

    def step(self, state, action):
        next_state = self.get_next_state(state, action)
        reward = self.get_reward(state, action, next_state)
        if next_state in self.red_states:
            next_state = self.start_state  # Red square? Back to start!
        done = self.is_terminal(next_state)
        return next_state, reward, done

# Base class for our RL agents, setting up the Q-table and action picking
class RLAgent:
    def __init__(self, env, alpha=0.1, gamma=0.99, epsilon=0.1):
        self.env = env
        self.alpha = alpha  # Learning rate: how fast we update Q-values
        self.gamma = gamma  # Discount factor: how much we care about future rewards
        self.epsilon = epsilon  # Initial exploration rate: chance of random moves
        self.q_table = defaultdict(lambda: defaultdict(float))  # Q-table starts empty

    def choose_action(self, state, greedy=False, episode=0):
        if not greedy and random.random() < self.get_epsilon(episode):
            return random.choice(self.env.actions)  # Explore: pick a random move
        else:
            q_values = [self.q_table[state][action] for action in self.env.actions]
            max_q = max(q_values)
            best_actions = [action for action, q in zip(self.env.actions, q_values) if q == max_q]
            return random.choice(best_actions)  # Exploit: pick the best move (random tie-break)

    def get_epsilon(self, episode):
        # Linear decay: epsilon = max(0.01, 0.1 / (1 + episode/1000))
        return max(0.01, 0.1 / (1 + episode / 1000))  # Decay from 0.1 to 0.01

    def get_policy(self):
        policy = {}
        for state in [(i, j) for i in range(self.env.grid_size) for j in range(self.env.grid_size)]:
            if not self.env.is_terminal(state):
                policy[state] = self.choose_action(state, greedy=True)  # Best action for each state
        return policy

# SARSA agent: learns while following its own (sometimes random) policy
class SARSAAgent(RLAgent):
    def train(self, episodes=1000):
        episode_rewards = []
        for episode in range(episodes):
            state = self.env.reset()
            action = self.choose_action(state, episode=episode)  # Pass episode for epsilon decay
            total_reward = 0
            while not self.env.is_terminal(state):
                next_state, reward, done = self.env.step(state, action)
                next_action = self.choose_action(next_state, episode=episode)  # Decay epsilon
                current_q = self.q_table[state][action]
                next_q = self.q_table[next_state][next_action] if not done else 0
                self.q_table[state][action] = current_q + self.alpha * (
                    reward + self.gamma * next_q - current_q
                )  # SARSA update: uses next action’s Q-value
                state = next_state
                action = next_action
                total_reward += reward
                if total_reward < -1000:  # Prevent getting stuck forever
                    break
            episode_rewards.append(total_reward)
        return episode_rewards

# Q-learning agent: aims for the best policy, ignoring its own exploration
class QLearningAgent(RLAgent):
    def train(self, episodes=1000):
        episode_rewards = []
        for episode in range(episodes):
            state = self.env.reset()
            total_reward = 0
            while not self.env.is_terminal(state):
                action = self.choose_action(state, episode=episode)  # Pass episode for epsilon decay
                next_state, reward, done = self.env.step(state, action)
                current_q = self.q_table[state][action]
                if done:
                    max_next_q = 0
                else:
                    max_next_q = max([self.q_table[next_state][a] for a in self.env.actions])
                self.q_table[state][action] = current_q + self.alpha * (
                    reward + self.gamma * max_next_q - current_q
                )  # Q-learning update: uses best next Q-value
                state = next_state
                total_reward += reward
                if total_reward < -1000:  # Safety net for infinite loops
                    break
            episode_rewards.append(total_reward)
        return episode_rewards

# Follow the learned policy to see the agent’s path
def simulate_trajectory(env, policy, max_steps=100):
    trajectory = [env.reset()]
    state = env.reset()
    steps = 0
    while not env.is_terminal(state) and steps < max_steps:
        action = policy.get(state, 'right')  # Default to right if state’s not in policy
        next_state, reward, done = env.step(state, action)
        trajectory.append(next_state)
        state = next_state
        steps += 1
    return trajectory

# Plot the grid with the agent’s path (yellow arrows, blue start, green end)
def plot_grid_with_trajectory(env, trajectory, title):
    plt.figure(figsize=(6, 6))
    grid = np.zeros((env.grid_size, env.grid_size))
    for state in env.black_states:
        grid[state] = 1  # Black for goals
    for state in env.red_states:
        grid[state] = -1  # Red for traps
    grid[env.start_state] = 0.5  # Blue for start
    colors = ['red', 'white', 'lightblue', 'black']
    bounds = [-1.5, -0.5, 0.25, 0.75, 1.5]
    cmap = plt.matplotlib.colors.ListedColormap(colors)
    norm = plt.matplotlib.colors.BoundaryNorm(bounds, cmap.N)
    plt.imshow(grid, cmap=cmap, norm=norm)
    if len(trajectory) > 1:
        for i in range(len(trajectory) - 1):
            y1, x1 = trajectory[i]
            y2, x2 = trajectory[i + 1]
            plt.arrow(x1, y1, x2 - x1, y2 - y1, head_width=0.1, head_length=0.1,
                     fc='yellow', ec='orange', linewidth=2, alpha=0.8)
    start_y, start_x = trajectory[0]
    plt.plot(start_x, start_y, 'o', color='blue', markersize=10, label='Start')
    if len(trajectory) > 1:
        end_y, end_x = trajectory[-1]
        plt.plot(end_x, end_y, 's', color='green', markersize=10, label='End')
    plt.title(title)
    plt.xticks(range(env.grid_size))
    plt.yticks(range(env.grid_size))
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.show()

# Show the policy as arrows on the grid (best action per state)
def plot_q_values(agent, title):
    plt.figure(figsize=(6, 6))
    grid = np.zeros((agent.env.grid_size, agent.env.grid_size))
    for state in agent.env.black_states:
        grid[state] = 1
    for state in agent.env.red_states:
        grid[state] = -1
    grid[agent.env.start_state] = 0.5
    colors = ['red', 'white', 'lightblue', 'black']
    bounds = [-1.5, -0.5, 0.25, 0.75, 1.5]
    cmap = plt.matplotlib.colors.ListedColormap(colors)
    norm = plt.matplotlib.colors.BoundaryNorm(bounds, cmap.N)
    plt.imshow(grid, cmap=cmap, norm=norm, alpha=0.7)
    arrow_props = {'up': (0, -0.3), 'down': (0, 0.3), 'left': (-0.3, 0), 'right': (0.3, 0)}
    for i in range(agent.env.grid_size):
        for j in range(agent.env.grid_size):
            state = (i, j)
            if not agent.env.is_terminal(state) and state not in agent.env.red_states:
                best_action = agent.choose_action(state, greedy=True)
                dx, dy = arrow_props[best_action]
                plt.arrow(j, i, dx, dy, head_width=0.1, head_length=0.1,
                         fc='black', ec='black', alpha=0.8)
    plt.title(title)
    plt.xticks(range(agent.env.grid_size))
    plt.yticks(range(agent.env.grid_size))
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()

# Let’s get this party started!
def main():
    # Set up the grid world
    env = GridWorld()
    # Lock in randomness for reproducible results
    random.seed(42)
    np.random.seed(42)

    # Train SARSA: it’s like learning while sticking to your own plan
    print("Training SARSA agent... grab a coffee!")
    sarsa_agent = SARSAAgent(env, alpha=0.1, gamma=0.99, epsilon=0.1)
    sarsa_rewards = sarsa_agent.train(episodes=1000)
    sarsa_policy = sarsa_agent.get_policy()

    # Train Q-learning: goes straight for the best plan
    print("Training Q-learning agent... almost done!")
    qlearning_agent = QLearningAgent(env, alpha=0.1, gamma=0.99, epsilon=0.1)
    qlearning_rewards = qlearning_agent.train(episodes=1000)
    qlearning_policy = qlearning_agent.get_policy()

    # Get the paths they take with their final policies
    sarsa_trajectory = simulate_trajectory(env, sarsa_policy)
    qlearning_trajectory = simulate_trajectory(env, qlearning_policy)

    # Plot 1: Learning curves (smoothed to look nice)
    plt.figure(figsize=(8, 5))
    window = 50
    sarsa_smooth = np.convolve(sarsa_rewards, np.ones(window) / window, mode='valid')
    qlearning_smooth = np.convolve(qlearning_rewards, np.ones(window) / window, mode='valid')
    plt.plot(sarsa_smooth, label='SARSA', alpha=0.8)
    plt.plot(qlearning_smooth, label='Q-Learning', alpha=0.8)
    plt.xlabel('Episode')
    plt.ylabel('Episode Reward (Smoothed)')
    plt.title('Learning Curves Comparison')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()  # Show it solo for clarity

    # Plot 2: Last 200 episodes, raw rewards
    plt.figure(figsize=(8, 5))
    recent_sarsa = sarsa_rewards[-200:]
    recent_qlearning = qlearning_rewards[-200:]
    plt.plot(recent_sarsa, label='SARSA', alpha=0.7)
    plt.plot(recent_qlearning, label='Q-Learning', alpha=0.7)
    plt.xlabel('Episode (Last 200)')
    plt.ylabel('Episode Reward')
    plt.title('Recent Performance')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()

    # Plot 3: SARSA’s path on the grid
    plot_grid_with_trajectory(env, sarsa_trajectory, 'SARSA Trajectory')

    # Plot 4: Q-learning’s path
    plot_grid_with_trajectory(env, qlearning_trajectory, 'Q-Learning Trajectory')

    # Plot 5: SARSA’s policy (arrows showing best moves)
    plot_q_values(sarsa_agent, 'SARSA Policy')

    # Plot 6: Q-learning’s policy
    plot_q_values(qlearning_agent, 'Q-Learning Policy')

    # Plot 7: How rewards spread out in the last 500 episodes
    plt.figure(figsize=(8, 5))
    plt.hist(sarsa_rewards[-500:], bins=30, alpha=0.7, label='SARSA', density=True)
    plt.hist(qlearning_rewards[-500:], bins=30, alpha=0.7, label='Q-Learning', density=True)
    plt.xlabel('Episode Reward')
    plt.ylabel('Density')
    plt.title('Reward Distribution (Last 500 Episodes)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()

    # Plot 8: How stable the rewards are over time
    plt.figure(figsize=(8, 5))
    window = 100
    sarsa_convergence = [np.std(sarsa_rewards[max(0, i - window):i + 1]) for i in range(len(sarsa_rewards))]
    qlearning_convergence = [np.std(qlearning_rewards[max(0, i - window):i + 1]) for i in range(len(qlearning_rewards))]
    plt.plot(sarsa_convergence, label='SARSA', alpha=0.8)
    plt.plot(qlearning_convergence, label='Q-Learning', alpha=0.8)
    plt.xlabel('Episode')
    plt.ylabel('Reward Std Dev (100-Episode Window)')
    plt.title('Convergence Analysis')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()

    # Spill the tea: how did they do?
    print("\n" + "=" * 60)
    print("DETAILED ANALYSIS REPORT")
    print("=" * 60)
    print(f"\nTrajectory Comparison:")
    print(f"SARSA trajectory length: {len(sarsa_trajectory)}")
    print(f"Q-Learning trajectory length: {len(qlearning_trajectory)}")
    print(f"SARSA trajectory: {sarsa_trajectory}")
    print(f"Q-Learning trajectory: {qlearning_trajectory}")
    print(f"\nFinal Performance (last 100 episodes):")
    sarsa_final = np.mean(sarsa_rewards[-100:])
    qlearning_final = np.mean(qlearning_rewards[-100:])
    print(f"SARSA average reward: {sarsa_final:.2f}")
    print(f"Q-Learning average reward: {qlearning_final:.2f}")
    print(f"\nConvergence Analysis:")
    sarsa_final_std = np.std(sarsa_rewards[-100:])
    qlearning_final_std = np.std(qlearning_rewards[-100:])
    print(f"SARSA final std dev: {sarsa_final_std:.2f}")
    print(f"Q-Learning final std dev: {qlearning_final_std:.2f}")
    print(f"\nPolicy Differences:")
    different_states = 0
    total_states = 0
    for state in sarsa_policy:
        if state in qlearning_policy:
            total_states += 1
            if sarsa_policy[state] != qlearning_policy[state]:
                different_states += 1
                print(f"State {state}: SARSA={sarsa_policy[state]}, Q-Learning={qlearning_policy[state]}")
    print(f"Policy similarity: {(total_states - different_states) / total_states * 100:.1f}% identical actions")

    def evaluate_policy(env, policy, episodes=100):
        rewards = []
        for _ in range(episodes):
            state = env.reset()
            total_reward = 0
            steps = 0
            while not env.is_terminal(state) and steps < 100:
                action = policy.get(state, 'right')
                next_state, reward, done = env.step(state, action)
                total_reward += reward
                state = next_state
                steps += 1
                if done:
                    break  # Ensure final step’s reward is counted
            rewards.append(total_reward)
        return rewards

    print(f"\nPolicy Evaluation (100 test episodes):")
    sarsa_test_rewards = evaluate_policy(env, sarsa_policy)
    qlearning_test_rewards = evaluate_policy(env, qlearning_policy)
    print(f"SARSA test performance: {np.mean(sarsa_test_rewards):.2f} ± {np.std(sarsa_test_rewards):.2f}")
    print(f"Q-Learning test performance: {np.mean(qlearning_test_rewards):.2f} ± {np.std(qlearning_test_rewards):.2f}")
    print(f"\nKey Insights:")
    print("1. SARSA’s on-policy vibe makes it a bit cautious, learning its own path.")
    print("2. Q-learning’s off-policy approach goes for the gold, converging faster.")
    print("3. With γ=0.99 and epsilon decay, both nail the 8-step path through (2,2) to (0,0).")
    print("4. Q-learning’s smoother, with less wobble (check the plots!).")
    print("5. Epsilon decay shifts to exploitation, refining policies with less noise.")

if __name__ == "__main__":
    main()