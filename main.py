"""Simulates k-armed bandits to demonstrate the average reward and
chance of forming the optimal policy after a certain number of
episodes

Colin Siles
Created 10 August 2018
"""

import random
import tqdm
import matplotlib.pyplot as plt
import numpy as np

# Constants
ARMS = 10		# Number of levers/arms for the enviornment
EPISODES = 1000 	# Number of steps agent is allowed to train
TRIALS = 2000 		# Number of trials to ensure accurate results
STATIONARY = True 	# Determines if situation is stationary or not

# Experiment Variables
EPSILONS = [0.0, 0.05, 0.1, 0.2, 0.5, 1] # Chance bandit makes a random move
INITIALS = [0]  # Initial value for the bandit's Q-table
ALPHAS = [None]	# Step-size for traning (None is sample-average method)

class Bandit(object):
	"""The agent. Manages choosing actions and training self."""

	def __init__(self, epsilon, initial, alpha = None):
		self.q_table = np.array([initial] * ARMS, dtype = 'float64')

		self.epsilon = epsilon

		if alpha == None:
			self.alpha = lambda action: (1/self.action_count[action])
		else:
			self.alpha = lambda action: alpha

		self.action_count = [0] * ARMS
		self.reward = []  # Tracks reward from each episode
		self.optimal = [] # Tracks 1 or 0 for if optimal action was chosen

	def choose_action(self):
		"""Returns action chosen based on q_table amd epsilon"""

		if not self.q_table.any(0) or random.random() < self.epsilon:
			action = random.randint(0, ARMS - 1)
		else:
			action = np.argmax(self.q_table)

		self.action_count[action] += 1

		return action

	def update_q_table(self, action, reward, steps, optimal):
		"""Trains the agent according to environment reward"""

		self.q_table[action] += self.alpha(action) * (reward - self.q_table[action])

		self.reward.append(reward)
		self.optimal.append(optimal)

class Levers(object):
	"""The environment. Handles providing rewards and other feedback"""

	def __init__(self, lever_num):
		"""Each lever stores the mean reward
		Optimal policy cached for speed"""

		self.levers = np.random.normal(0, 1, lever_num)
		self.optimal = np.argmax(self.levers)

	def get_reward(self, action):
		"""Returns a random reward based on the lever"""

		return 	random.gauss(self.levers[action], 1)

	def check_optimal(self, q_table):
		"""Determines if a q_table has determined the best lever.
		Does not test action directly since such a test would be
		biased against bandits with higher epsilons"""

		return int(np.argmax(q_table) == self.optimal)

	def change_levers(self):
		"""Changes the levers by a small amount to simulate a
		non-stationary problem"""

		self.levers += np.random.normal(0, 0.01, len(self.levers))
		self.optimal = np.argmax(self.levers)


def train(epsilon, alpha, initial, arms, episodes, trials):
	"""Main training loop for one test"""

	average_reward = np.array([0.0] * episodes)
	optimal_percent = np.array([0.0] * episodes)

	# Start at trial 1, episode 1 to avoid division by 0
	for trial in tqdm.trange(1, trials + 1):
		bandit = Bandit(epsilon, initial, alpha)
		levers = Levers(arms)

		for steps in range(1, episodes + 1):
			action = bandit.choose_action()
			reward = levers.get_reward(action)
			optimal = levers.check_optimal(bandit.q_table)

			bandit.update_q_table(action, reward, steps, optimal)

			if not STATIONARY:
				levers.change_levers()

		# Converted to numpy arrays after for speed
		new_reward = np.array(bandit.reward)
		new_optimal = np.array(bandit.optimal)

		# Performs sample-average update of arrays
		average_reward += (1/trial) * (new_reward - average_reward)
		optimal_percent += (1/trial) * (new_optimal - optimal_percent)

	# Plot the graphs, but don't display yet
	plt.figure(1)
	plt.plot(average_reward, label = build_label(epsilon, alpha, initial))

	plt.figure(2)
	plt.plot(optimal_percent, label = build_label(epsilon, alpha, initial))

def build_label(epsilon, alpha, initial):
	"""Adds information to label based on if it is unique"""

	label = ''

	if len(EPSILONS) != 1:
		label += 'E:{},'.format(epsilon)
	if len(ALPHAS) != 1:
		label += 'A:{},'.format(alpha)
	if len(INITIALS) != 1:
		label += 'I:{},'.format(initial)

	return label

def build_tests():
	"""Combines lists of test variables into a single list of tests"""

	tests = []
	for alpha in ALPHAS:
		for epsilon in EPSILONS:
			for initial in INITIALS:
				tests.append((epsilon, alpha, initial))

	return tests

def display_graphs():
	"""Finally displays collected data, including title and legend"""

	plt.figure(1)
	plt.legend(loc='best')
	plt.title('Step Number vs. Average Reward')

	plt.figure(2)
	plt.legend(loc='best')
	plt.title('Step Number vs. Optimal Policy Probability')

	plt.show()

def main():
	tests = build_tests()

	for test in tests:
		epsilon, alpha, initial = test
		train(epsilon, alpha, initial, ARMS, EPISODES, TRIALS)

	display_graphs()

if __name__ == '__main__':
	main()
