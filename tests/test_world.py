import pytest
import sys
import os
import numpy as np
sys.path.append(os.path.dirname(os.path.realpath(__file__)) + "/../gym_grid")
from gym_grid.envs.world import World



# @pytest.mark.parametrize("size_x, size_y",[(1, 1), (10,10)])
# def test_generate_grid_filled_with_0(world, size_x, size_y):
#     world.generate_grid_with_size(size_x, size_y)
#     result = world.get_grid()
#     assert result.shape == (size_x, size_y)
#     assert np.array_equal(result,np.zeros((size_x, size_y)))

# def test_step():
#     world.step(action)

@pytest.fixture
def world():
    world = World()
    yield world

def test_empty_grid_with_size_1by1(world):
    world.create_world((1,1))
    result = world.get_grid()
    assert np.array_equal(result, np.zeros((1,1)))


def test_add_one_entity(world):
    world.create_world((1,3))
    world.add_entity()
    result = world.get_num_of_entity()
    assert result == 1

def test_add_entity_with_specific_point(world):
    world.create_world((3,3))
    world.add_entity((2,1))
    assert world.get_pos() == (2,1)

def test_set_entity_pos(world):
    world.create_world((3,3))
    world.add_entity((2,1))
    world.set_pos((1,2))
    assert world.get_pos() == (1,2)

def test_get_pos_of_one_entity_at_init_point(world):
    world.create_world((3,3))
    world.add_entity()
    assert world.get_pos() == (0, 0)


def test_move_one_entity(world):
    world.create_world((3,3))
    world.add_entity()
    assert world.get_pos() == (0, 0)
    world.move((1,0))  # Right
    assert world.get_pos() == (1, 0)
    world.move((0,1))  # Up
    assert world.get_pos() == (1, 1)
    world.move((-1,0))  # Left
    assert world.get_pos() == (0, 1)
    world.move((0,-1))  # Down
    assert world.get_pos() == (0, 0)

def test_check_boundary(world):
    world.create_world((3,3))
    world.add_entity()

    world.move((-1,0))  # Left
    assert world.get_pos() == (0, 0)
    world.move((0,-1))  # Down
    assert world.get_pos() == (0, 0)

    world.move((1,0))  # Right
    world.move((1,0))  # Right
    world.move((1,0))  # Right    
    assert world.get_pos() == (2, 0)
    world.move((0,1))  # Up
    world.move((0,1))  # Up
    world.move((0,1))  # Up
    world.move((0,1))  # Up
    assert world.get_pos() == (2, 2)







