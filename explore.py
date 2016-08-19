import numpy as np
import matplotlib.pyplot as plt
from maze import example
from agent import Agent


class Task(object):

    def __init__(self, agent, maze):
        agent.initialize(maze)
        self.agent = agent
        self.maze = maze

    def explore(self):
        i = 0
        while True:
            next_position = self.agent.epsilon_greedy(self.maze)
            reward = self.maze.get_reward(next_position)
            self.agent.update_value(reward, next_position)
            self.agent.move(next_position)
            i += 1
            if tuple(self.agent.position) == tuple(self.maze.goal):
                break
        self.agent.restart(self.maze)
        return i

    def solve(self):
        i = 0
        while True:
            next_position = self.agent.solve(self.maze)
            self.agent.move(next_position)
            i += 1
            if tuple(self.agent.position) == tuple(self.maze.goal):
                break
        self.agent.restart(self.maze)
        return i


def perform_task():
    agent = Agent(epsilon=0.1, gamma=0.95, alpha=0.1)
    task = Task(agent, example)
    n_moves = []

    for step in xrange(1000):
        task.explore()
        i = task.solve()
        print "step %5d, %d" % (step, i)
        n_moves.append(i)

    agent.draw_values()
    plt.plot(np.arange(len(n_moves)), np.array(n_moves))
    plt.show()


if __name__ == '__main__':
    perform_task()
