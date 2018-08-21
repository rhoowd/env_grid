from __future__ import division

import pytest
import sys
import os
import numpy as np
sys.path.append(os.path.dirname(os.path.realpath(__file__)) + "../gym_grid")
from gym_grid.envs.dsdp import DsdpEnv


@pytest.fixture
def env():
    env = DsdpEnv()
    yield env


def test_init_state(env):
    assert env.get_state() == ((1,0), False)  # Checkt init state

def test_gym():
    import gym
    import gym_grid

    env = gym.make('grid-dsdp-v1')

    obs = env.reset()
    print(obs)
    assert np.array_equal(obs, env.one_hot(1))
    
    env.render()
    o, r, d, i = env.step(1)
    print(o, r, d, i)
    env.render()
    assert d

    env.step(1)
    print(env.action_space.sample())


