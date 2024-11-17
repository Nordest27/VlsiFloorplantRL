# Example from:
# https://stackoverflow.com/questions/44469266/how-to-implement-custom-environment-in-keras-rl-openai-gym
# https://github.com/openai/gym/blob/pr/429/gym/envs/toy_text/hotter_colder.py
import random
import string
from typing import Optional, Tuple

import gym
from gym import spaces
from gym.core import ObsType
from gym.utils import seeding
import numpy as np

from copy import copy

from vlsi_floorplant import PyFloorPlantProblem
def to_numpy_array(l: list[int]):
    return np.asarray([np.float32(v) for v in l])


class FloorPlantEnv(gym.Env):

    fpp: PyFloorPlantProblem = None
    sa_fpp: PyFloorPlantProblem = None
    best_fpp: PyFloorPlantProblem = None
    best_obj: int = -np.inf
    n: int

    def __init__(self, n: int):
        self.n = n
        self.action_space = spaces.Tuple([
            spaces.Discrete(2*n),
        ])
        self.observation = None
        self.reset()
        super().__init__()

    def reset(self):
        if not self.fpp:
            self.fpp = PyFloorPlantProblem(self.n)
            self.sa_fpp = self.fpp.copy()
            self.sa_fpp.apply_simulated_annealing(100, 1.0-1e-5)
            self.best_fpp = self.fpp.copy()
        self.fpp.shuffle_sp()
        self.observation = tuple([
            [], []
        ])

    def get_fpp_input(self) -> np.array:
        return [*self.fpp.x(), *self.fpp.y()]

    def get_input(self) -> np.ndarray:
        # Assuming offsets is a flat array combining two observations
        x = self.observation[0] + [self.n]*(self.n-len(self.observation[0]))
        y = self.observation[1] + [self.n]*(self.n-len(self.observation[1]))
        return np.asarray([*x, *y])

    def action(self, value: int, is_y: bool) -> tuple[float, bool]:
        new_fpp = self.fpp.copy()
        #new_fpp.set_sp(self.observation[0], self.observation[1])
        new_fpp.apply_simulated_annealing(1, 1.0 - 1e-3)
        obj = -new_fpp.get_current_sp_objective()
        if obj > self.best_obj:
            self.best_obj = obj
            self.best_fpp = new_fpp
        return obj/100, True
        """
        if value >= self.n or value < 0:
            return -1, False
        if is_y:
            for y in self.observation[1]:
                if y == value:
                    return -1, False
            self.observation[1].append(value)
        else:
            for x in self.observation[0]:
                if x == value:
                    return -1, False
            self.observation[0].append(value)

        if len(self.observation[0]) == self.n and len(self.observation[1]) == self.n:
            new_fpp = self.fpp.copy()
            new_fpp.set_sp(self.observation[0], self.observation[1])
            new_fpp.apply_simulated_annealing(1, 1.0 - 1e-3)
            obj = -new_fpp.get_current_sp_objective()
            if obj > self.best_obj:
                self.best_obj = obj
                self.fpp = new_fpp
            return obj/100, True

        return 0, False
        """

    def render(self):
        self.fpp.visualize()

    def visualize_sa_solution(self):
        print("Sa solution objective: ", self.sa_fpp.get_current_sp_objective())
        self.sa_fpp.visualize()
