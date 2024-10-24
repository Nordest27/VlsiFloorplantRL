# Example from:
# https://stackoverflow.com/questions/44469266/how-to-implement-custom-environment-in-keras-rl-openai-gym
# https://github.com/openai/gym/blob/pr/429/gym/envs/toy_text/hotter_colder.py
import random
import string

import gym
from gym import spaces
from gym.utils import seeding
import numpy as np

from copy import copy

from vlsi_floorplant import PyFloorPlantProblem
def to_numpy_array(l: list[int]):
    return np.asarray([np.float32(v) for v in l])

def to_positions(l: list[int]) -> list[int]:
    positions = copy(l)
    for i, v in enumerate(l):
        positions[v] = i
    return positions

def position_connected_to_position(
    connected_to: tuple[tuple[int, ...], ...],
    vec: list[int]
) ->  tuple[tuple[int, ...], ...]:
    range_n = range(len(vec))
    return tuple([tuple([int(connected_to[vec[i]][vec[j]]) for i in range_n]) for j in range_n])


class FloorPlantEnv(gym.Env):

    fpp: PyFloorPlantProblem = None
    best_fpp: PyFloorPlantProblem = None
    current_fpp: PyFloorPlantProblem = None
    n: int
    best_obj: int = -1
    previous_obj: int = -1
    current_obj: int = -1
    obj: int = -1
    max_steps: int = 5
    steps: int = 0
    since_previous_upgrade: int = 0

    def __init__(self, n: int):
        self.n = n
        self.action_space = spaces.Tuple(
            spaces=[
                # Which first
                spaces.Discrete(self.n),
                # Which second
                spaces.Discrete(self.n),
                # Type of move
                spaces.Discrete(10)
            ]
        )
        self.observation_space=spaces.Tuple([
            # offset widths
            # spaces.Box(low=0, high=np.inf, shape=(n,)),
            # offset heights
            # spaces.Box(low=0, high=np.inf, shape=(n,)),
            # positions connected to positions X
            spaces.Tuple([
                spaces.Tuple([
                    spaces.Discrete(2)
                    for _ in range(self.n)])
                for _ in range(self.n)]),
            # positions connected to positions Y
            spaces.Tuple([
                spaces.Tuple([
                    spaces.Discrete(2)
                    for _ in range(self.n)])
                for _ in range(self.n)])
        ])
        self.seed()
        self.reset()
        super().__init__()

    def get_input(self) -> tuple[np.ndarray, np.ndarray]:
        # Assuming offsets is a flat array combining two observations
        #offsets = to_numpy_array(list(self.observation[0]) + list(self.observation[1]))

        # Convert x_con and y_con to arrays and reshape them for convolutional layers
        x_con = np.array([to_numpy_array(x_con_row) for x_con_row in self.observation[0]])
        y_con = np.array([to_numpy_array(y_con_row) for y_con_row in self.observation[1]])

        # Reshape to add a channel dimension
        x_con = x_con.reshape((x_con.shape[0], x_con.shape[1], 1))  # (batch_size, height, width, channels)
        y_con = y_con.reshape((y_con.shape[0], y_con.shape[1], 1))  # (batch_size, height, width, channels)

        return x_con, y_con

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def reset(self):

        if not self.best_fpp:
            self.best_fpp = PyFloorPlantProblem(self.n)
            self.current_fpp = self.best_fpp.copy()
            self.fpp = self.best_fpp.copy()

        if self.best_obj < self.obj:
            self.best_fpp = self.fpp.copy()

        self.fpp = self.current_fpp.copy()
        self.fpp.shuffle_sp()

        self.best_obj = -self.best_fpp.get_current_sp_objective()
        self.current_obj = -self.current_fpp.get_current_sp_objective()
        self.obj = -self.fpp.get_current_sp_objective()

        self.steps = 0

        connected_to = tuple([tuple([int(v) for v in row]) for row in self.fpp.connected_to()])
        self.observation = tuple([
            position_connected_to_position(connected_to, self.fpp.x()),
            position_connected_to_position(connected_to, self.fpp.y()),
        ])
        self.since_previous_upgrade += 1
        assert self.observation_space.contains(self.observation)

    def think_step(self, action: tuple[int, int, int]):
        assert self.action_space.contains(action)
        #print("Taking action: ", action)
        i, j, move = action
        if self.fpp.x()[0] == i or self.fpp.y()[0] == j:
            return self.observation, 100, True, {}
        else:
            return self.observation, 0, True, {}

        self.steps += 1
        if self.steps > self.max_steps:
            return self.observation, 0, True, {}

        if i >= self.n or j >= self.n or i == j:
            print("SHOULDNT HAPPEN!")
            return self.observation, 0, False, {}

        self.previous_obj = self.obj
        self.obj = -self.fpp.apply_sp_move(i, j, move)

        connected_to = tuple([tuple([int(v) for v in row]) for row in self.fpp.connected_to()])
        self.observation = tuple([
            position_connected_to_position(connected_to, self.fpp.x()),
            position_connected_to_position(connected_to, self.fpp.y()),
        ])
        assert self.observation_space.contains(self.observation)

        return self.observation, self.obj - self.current_obj, False, {}

    def step(self, action: tuple[int, int, int]):
        assert self.action_space.contains(action)
        #print("Taking action: ", action)
        i, j, move = action

        self.steps += 1
        if self.steps > self.max_steps:
            return self.observation, 0, True, {}

        if i >= self.n or j >= self.n or i == j:
            return self.observation, 0, False, {}

        self.current_fpp.apply_sp_move(i, j, move)

        return self.observation, 0, False, {}

    def render(self):
        self.fpp.visualize()

if __name__ == "__main__":
    fpe = FloorPlantEnv(10)
    print(fpe.fpp.connected_to)
    print(fpe.observation)
    print(fpe.fpp.visualize())
    print(fpe.step((6, 9, 2))[1])
    print(fpe.fpp.visualize())
    print(fpe.step((0, 1, 9))[1])
    print(fpe.fpp.visualize())



