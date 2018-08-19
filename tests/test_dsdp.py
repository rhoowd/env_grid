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

def test_deterministic_action(env):
    take_deterministic_actions(1, env)  # "Right" action
    assert env.get_state() == ((2,0), False) 
    take_deterministic_actions(-1, env)  # "Left" action
    assert env.get_state() == ((1,0), False)  # Pos, Flag for s6
    take_deterministic_actions(3, env)  # "Right"
    assert env.get_state() == ((4,0), False) 
    take_deterministic_actions(1, env)  # "Right" action
    assert env.get_state() == ((5,0), True) # Change flag to True after visiting (5,0)
    take_deterministic_actions(-1, env)  # "Left" action
    assert env.get_state() == ((4,0), True) 
    assert env.get_obs() == 4

def test_terminate(env):
    assert not env.check_done()
    take_deterministic_actions(-1, env)  # "Left"
    assert env.check_done()

def test_reset(env):
    take_deterministic_actions(5, env)  # "Right"
    assert env.get_state() == ((5,0), True) 
    result = env.reset()    
    assert result == 1  # Checkt return of reset()
    assert env.get_state() == ((1,0), False)  # Checkt init state

def test_reward(env):
    assert env.get_reward() == 0    
    take_deterministic_actions(-1, env)  # "Left"
    assert env.get_reward() == 0.01    
    env.reset()
    take_deterministic_actions(5, env)  # "Right" 5 times
    assert env.get_reward() == 0    
    take_deterministic_actions(-5, env)  # "Left" 5 times
    assert env.get_reward() == 1

def test_action_type_1(env):
    trial = 100
    cnt_0 = 0
    cnt_2 = 0
    for _ in range(trial):
        env.reset()
        env.take_action(0)
        obs = env.get_obs()

        if obs == 0:
            cnt_0 += 1
        elif obs == 2:
            cnt_2 += 1
            
    assert cnt_0/trial > 0.4  # It should be 0.5 as trial increases
    assert cnt_2/trial > 0.4


def take_deterministic_actions(num, env):
    for _ in range(abs(num)):
        if num > 0:
            env.take_action(0, a_type=1)  # "Right"
        else:
            env.take_action(1, a_type=1)  # "Left"
    

def test_step(env):
    env.render()
    env.step(0)
    env.render()
    env.step(1)
    env.render()


def test_gym():
    import gym
    import gym_grid

    env = gym.make('grid-dsdp-v0')

    obs = env.reset()
    assert obs == 1  # Checkt init state
    env.render()
    o, r, d, i = env.step(1)
    print(o, r, d, i)
    env.render()
    assert d


