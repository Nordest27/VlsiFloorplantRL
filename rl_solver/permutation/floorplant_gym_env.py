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


class FloorPlantEnv(gym.Env):

    fpp: PyFloorPlantProblem = None
    best_obj: int = -1
    n: int

    def __init__(self, n: int):
        self.n = n
        self.action_space = spaces.Tuple([
            spaces.Discrete(i)
            for i in range(2, n+1)
        ])
        self.observation_space=spaces.Tuple([
            # offset widths
            spaces.Box(low=0, high=np.inf, shape=(n,)),
            # offset heights
            spaces.Box(low=0, high=np.inf, shape=(n,)),
            # positions connected to positions X
            spaces.Tuple([
                spaces.Tuple([
                    spaces.Discrete(2)
                    for _ in range(self.n)])
                for _ in range(self.n)
            ])
        ])
        self.fpp = PyFloorPlantProblem(self.n)
        connected_to = tuple([tuple([int(v) for v in row]) for row in self.fpp.connected_to()])
        self.observation = tuple([
            to_numpy_array(self.fpp.widths()),
            to_numpy_array(self.fpp.heights()),
            connected_to,
        ])
        assert self.observation_space.contains(self.observation)
        super().__init__()

    def get_input(self) -> tuple[np.ndarray, np.ndarray]:
        # Assuming offsets is a flat array combining two observations
        dims = to_numpy_array(list(self.observation[0]) + list(self.observation[1]))

        # Convert x_con and y_con to arrays and reshape them for convolutional layers
        conn = np.array([to_numpy_array(x_con_row) for x_con_row in self.observation[2]])

        # Reshape to add a channel dimension
        conn = conn.reshape((conn.shape[0], conn.shape[1], 1))  # (batch_size, height, width, channels)

        return dims, conn

    def get_permutation(self, x: list[int], y: list[int]) -> float:
        assert self.action_space.contains((tuple(x), tuple(y)))
        #print("Taking action: ", action)

        new_fpp = self.fpp.copy()
        new_fpp.set_sp(x, y)

        obj = -new_fpp.get_objective()
        if obj > self.best_obj:
            self.best_obj = obj
            self.fpp = new_fpp

        return -new_fpp.get_current_sp_objective()

    def render(self):
        self.fpp.visualize()
