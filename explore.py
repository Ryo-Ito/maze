import random
import numpy as np
import matplotlib.pyplot as plt


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

    def select_action(self, maze):
        # epsilon greedy
        candidates = maze.reachable_places(self.position)
        n = len(candidates)
        if self.epsilon < random.random():
            return candidates[np.random.randint(n)]
        else:
            values = np.ones(n) * self.gamma
            for i, place in enumerate(candidates):
                values[i] *= self.values[tuple(place)]
                values[i] += maze.get_reward(place)
            if np.abs(np.min(values) - np.max(values)) < 1e-5:
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
        plt.imshow(self.values, cmap='hot', interpolation='nearest')
        plt.show()


class Maze(object):

    def __init__(self, structure, start, goal):
        self.structure = structure
        self.start = np.array(start)
        self.goal = np.array(goal)
        self.shape = structure.shape

    def get_reward(self, position):
        if tuple(position) == tuple(self.goal):
            return 1.
        else:
            return 0.

    def reachable_places(self, position):
        actions = np.array([[1, 0], [0, 1], [-1, 0], [0, -1]])
        candidates = actions + np.array(position)
        mask = np.ones(4, dtype=bool)
        for i, place in enumerate(candidates):
            if self.structure[tuple(place)] == 1:
                mask[i] = False
        reachable = candidates[mask]
        assert len(reachable) > 0
        return reachable

    def draw(self):
        ax = plt.axes()
        ax.xaxis.set_ticklabels([])
        ax.yaxis.set_ticklabels([])
        plt.imshow((1 - self.structure),
                   cmap=plt.cm.gray,
                   interpolation='nearest',
                   alpha=0.5)
        plt.xticks(np.arange(self.shape[0]) + 0.5)
        plt.yticks(np.arange(self.shape[1]) + 0.5)
        plt.grid(ls='solid')
        plt.show()


def main():
    structure = np.array([[1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
                          [1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1],
                          [1, 0, 0, 1, 0, 1, 0, 0, 1, 1, 0, 1],
                          [1, 1, 0, 1, 0, 1, 1, 1, 1, 0, 0, 1],
                          [1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1],
                          [1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1],
                          [1, 1, 1, 1, 1, 0, 1, 0, 1, 0, 0, 1],
                          [1, 0, 0, 0, 0, 0, 1, 0, 1, 0, 0, 1],
                          [1, 0, 1, 1, 1, 1, 1, 0, 1, 0, 0, 1],
                          [1, 0, 0, 0, 0, 1, 0, 0, 1, 0, 0, 1],
                          [1, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 1],
                          [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]])

    maze = Maze(structure, start=(1, 1), goal=(10, 10))
    # maze.draw()

    agent = Agent(epsilon=0.01, gamma=0.95, alpha=0.25)
    agent.initialize(maze)

    steps = []

    for _ in xrange(100000):
        i = 0
        while True:
            next_position = agent.select_action(maze)
            reward = maze.get_reward(next_position)
            agent.update_value(reward, next_position)
            agent.move(next_position)
            i += 1
            if tuple(agent.position) == tuple(maze.goal):
                break
        agent.restart(maze)
        print "step %5d, %d" % (_, i)
        steps.append(i)

    agent.draw_values()
    plt.plot(np.arange(len(steps)), np.array(steps))
    plt.show()


if __name__ == '__main__':
    main()
