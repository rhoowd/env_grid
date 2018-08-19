from __future__ import print_function
from __future__ import division
import numpy as np
from world import World


class DsdpEnv(object):
    def __init__(self):
        self.world = World()
        self.world.create_world((6,1))
        self.world.add_entity((1,0))
        self.flag = False

    def reset(self):
        self.world.set_pos((1,0))
        self.flag = False
        return self.get_obs()

    def step(self, action):
        self.take_action(action)
        obs = self.get_obs()
        reward = self.get_reward()
        done = self.check_done()
        return obs, reward, done, None

    def render(self):
        for i in range(self.world.grid_width):
            print(int(self.world.grid[i][0]), end='')
        print("")

    def take_action(self, action, a_type=0):
        assert a_type < 2
        assert action in [0, 1]

        if a_type == 0:  # Default action for h-DQN (stochastic for "right" action)
            if action == 0:  # "Right"
                dx = np.random.choice([1, -1], 1, p=[0.5, 0.5])[0] 
                direction = (dx, 0)
            elif action == 1:  # "Left"
                direction = (-1,0)

        elif a_type == 1:  # Action type 1 (deterministic)
            if action == 0:  # "Right"
                direction = (1,0)
            elif action == 1: # "Left"
                direction = (-1,0)

        self.world.move(direction)
        
        if self.get_state() == ((5, 0), False):
            self.flag = True

    def check_done(self):
        return self.get_obs() == (0,0)

    def get_reward(self):
        if self.check_done():
            if self.flag:
                return 1
            else:
                return 0.01
        else:
            return 0


    def get_state(self):
        return self.get_obs(), self.flag

    def get_obs(self):
        return self.world.get_pos()
