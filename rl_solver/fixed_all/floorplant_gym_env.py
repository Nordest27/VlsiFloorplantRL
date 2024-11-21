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

def separate_digits(i: int) -> tuple[int, ...]:
    return tuple(int(d) for d in str(i))

def join_digits(i: tuple[int]) -> int:
    return int("".join(map(str, i)))

def to_one_hot_encoding(t: tuple[int]) -> list[list[int]]:
    return [[int(v==i) for i in range(len(t))] for v in  t]

def to_positions(l: list[int]) -> list[int]:
    positions = copy(l)
    for i, v in enumerate(l):
        positions[v] = i
    return positions

class FloorPlantEnv(gym.Env):

    ini_fpp = PyFloorPlantProblem = None
    fpp: PyFloorPlantProblem = None
    best_fpp: PyFloorPlantProblem = None

    rand_ini_fpp: PyFloorPlantProblem = None
    rand_fpp: PyFloorPlantProblem = None
    rand_best_fpp: PyFloorPlantProblem = None

    best_obj: float
    ini_obj: float
    previous_obj: float

    n: int
    max_steps: int = 5
    steps: int = 0

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
            # X
            spaces.Tuple([spaces.Discrete(self.n) for _ in range(self.n)]),
            # Y
            spaces.Tuple([spaces.Discrete(self.n) for _ in range(self.n)]),
        ])
        self.reset()
        super().__init__()

    def flattened_observation(self) -> np.array:
        observation = []
        for ob in self.observation:
            observation.extend(np.ndarray.flatten(np.array(list(ob))))
        return np.array(observation)


    def reset(self):
        if not self.fpp:
            self.ini_fpp = PyFloorPlantProblem(self.n)
            self.best_fpp = self.ini_fpp.copy()
            self.best_obj = self.best_fpp.get_current_sp_objective()
            self.ini_obj = self.best_obj
            self.fpp = self.best_fpp.copy()

            self.sa_fpp = self.fpp.copy()
            print("Simulated Annealing...")
            self.sa_fpp.apply_simulated_annealing(100, 1.0-1e-4)
            print("Simulated Annealing result: ", self.sa_fpp.get_current_sp_objective())
            self.sa_fpp.visualize()

            self.rand_ini_fpp = self.fpp.copy()
            self.rand_best_fpp = self.fpp.copy()

        self.fpp = self.ini_fpp.copy()
        self.rand_fpp = self.rand_ini_fpp.copy()

#         self.fpp.shuffle_sp()
#         self.rand_fpp.shuffle_sp()
        self.previous_obj = self.fpp.get_current_sp_objective()

        self.steps = 0

        self.observation = tuple([
            tuple((self.fpp.x())),
            tuple((self.fpp.y())),
        ])
        assert self.observation_space.contains(self.observation)

    def step(self, action: tuple[int, int, int], just_step: bool = False):
        assert self.action_space.contains(action)
        i, j, move = action
        self.steps += 1

        if i >= self.n or j >= self.n or i == j:
            print("Stop!")
            print(i, j)
            return self.observation, -1, False, {}

        previous_obj = self.fpp.get_current_sp_objective()
        if move < 9:
            self.fpp.apply_sp_move(i, j, move)

            first_choice, second_choice = np.random.choice(self.n, 2, replace=False)
            self.rand_fpp.apply_sp_move(first_choice, second_choice, random.randint(0, 8))
        # elif move == 9:
        if not just_step:
            pass
            self.fpp.apply_simulated_annealing(0.11, 1.0-1e-3)
            self.rand_fpp.apply_simulated_annealing(0.11, 1.0-1e-3)

        obj = self.fpp.get_current_sp_objective()
        rand_obj = self.rand_fpp.get_current_sp_objective()

        if obj < self.best_obj:
           self.best_fpp = self.fpp.copy()
           self.best_obj = obj

        if rand_obj < self.rand_best_fpp.get_current_sp_objective():
            self.rand_best_fpp = self.rand_fpp.copy()

        self.observation = tuple([
            tuple((self.fpp.x())),
            tuple((self.fpp.y())),
        ])
        assert self.observation_space.contains(self.observation)
        return self.observation, (previous_obj-obj)/self.ini_obj, move == 9 or self.steps > self.max_steps, {}

    def render(self):
        self.fpp.visualize()