import random
import numpy as np
import seaborn as sns


class Agent(object):

    def __init__(self, epsilon, gamma, alpha):
        self.epsilon = epsilon
        self.gamma = gamma
        self.alpha = alpha

    def initialize(self, maze):
        self.position = maze.start
        self.values = np.zeros_like(maze.structure, dtype=np.float)

    def restart(self, maze):
        self.position = maze.start

    def move(self, position):
        self.position = position

    def epsilon_greedy(self, maze):
        # epsilon greedy
        candidates = maze.reachable_places(self.position)
        n = len(candidates)
        if random.random() < self.epsilon:
            return candidates[np.random.randint(n)]
        else:
            values = np.ones(n) * self.gamma
            for i, place in enumerate(candidates):
                values[i] *= self.values[tuple(place)]
                values[i] += maze.get_reward(place)
            if np.abs(np.min(values) - np.max(values)) < 1e-7:
                return candidates[np.random.randint(n)]
            else:
                index = np.argmax(values)
                return candidates[index]

    def solve(self, maze, epsilon=0.01):
        candidates = maze.reachable_places(self.position)
        n = len(candidates)
        if random.random() < epsilon:
            return candidates[np.random.randint(n)]
        else:
            values = np.ones(n) * self.gamma
            for i, place in enumerate(candidates):
                values[i] *= self.values[tuple(place)]
                values[i] += maze.get_reward(place)
            if np.abs(np.min(values) - np.max(values)) < 1e-7:
                return candidates[np.random.randint(n)]
            else:
                index = np.argmax(values)
                return candidates[index]

    def update_value(self, reward, next_position):
        TDerror = (reward
                   + self.gamma * self.values[tuple(next_position)]
                   - self.values[tuple(self.position)])
        self.values[tuple(self.position)] += self.alpha * TDerror

    def draw_values(self):
        # plt.imshow(self.values, cmap='hot', interpolation='nearest')
        # plt.show()
        sns.heatmap(self.values, annot=True, cmap="YlGnBu", fmt='.1f')
        sns.plt.show()
