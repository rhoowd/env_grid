from __future__ import print_function
from __future__ import division
import numpy as np
from world import World
import gym
from gym import spaces


class DsdpEnv(gym.Env):
    def __init__(self):
        self.size = 6
        self.world = World()
        self.world.create_world((self.size,1))
        self.world.add_entity((1,0))
        self.flag = False

        self.observation_space = spaces.Discrete(self.size)
        self.action_space = spaces.Discrete(2)  # 2 actions: 0 (Right), 1 (Left)

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
        assert self.action_space.contains(action)
        # assert action in [0, 1]

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
        return self.world.get_pos() == (0,0)

    def get_reward(self):
        if self.check_done():
            if self.flag:
                return 1
            else:
                return 0.01
        else:
            return 0


    def get_state(self):
        return self.world.get_pos(), self.flag

    def get_obs(self):
        return self.world.get_pos()[0]


class DsdpEnvOneHot(DsdpEnv):
    def get_obs(self):
        o = self.world.get_pos()[0]
        ret = self.one_hot(o)
        return ret

    def one_hot(self, o):
        ret = np.zeros(self.size)
        ret[o] = 1
        return ret

