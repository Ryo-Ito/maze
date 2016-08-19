import numpy as np
import matplotlib.pyplot as plt


class Maze(object):

    def __init__(self, structure, start, goal):
        self.structure = structure
        self.start = np.array(start)
        self.goal = np.array(goal)
        self.shape = structure.shape

    def get_reward(self, position):
        if tuple(position) == tuple(self.goal):
            return 100.
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
example = Maze(structure, (1, 1), (10, 10))
