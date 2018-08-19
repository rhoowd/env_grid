from __future__ import print_function
import numpy as np


class World(object):

    def __init__(self):
        self.grid = None
        self.grid_width = 0
        self.grid_height = 0

    def create_world(self, size):
        self.grid = np.zeros(size)
        self.grid_width = size[0]
        self.grid_height = size[1]

    def get_grid(self):
        return self.grid

    def add_entity(self, pos=(0,0)):
        self.grid[pos[0]][pos[1]] += 1
    
    def get_num_of_entity(self):
        return np.sum(self.grid)

    def get_pos(self):
        for i in range(self.grid_width):
            for j in range(self.grid_height):
                if self.grid[i][j] == 1:
                    return (i, j)
        assert False  # There is no entity

    def set_pos(self, pos):
        c_x, c_y = self.get_pos()
        self.grid[c_x][c_y] -= 1
        self.grid[pos[0]][pos[1]] += 1


    def move(self, direction):
        c_x, c_y = self.get_pos()
        n_x = min(max(c_x + direction[0], 0), self.grid_width-1)
        n_y = min(max(c_y + direction[1], 0), self.grid_height-1)
        self.grid[c_x][c_y] -= 1
        self.grid[n_x][n_y] += 1


