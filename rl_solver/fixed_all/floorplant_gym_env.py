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

    fpp: PyFloorPlantProblem = None
    best_sp: PyFloorPlantProblem = None
    n: int
    best_obj: int = -1
    previous_obj: int = -1
    obj: int = -1
    max_steps: int = 50
    steps: int = 0
    sa_temp: float = 100
    sa_alpha: float = 0.99

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
        self.seed()
        self.reset()
        super().__init__()

    def to_one_hot_encoding(self, t: tuple[int]) -> list[list[int]]:
        return [[int(v==i) for i in range(self.n)] for v in  t]

    def flattened_observation(self) -> np.array:
        observation = []
        for ob in self.observation:
            observation.extend(np.ndarray.flatten(np.array(list(ob))))
        return np.array(observation)

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def reset(self):

        if not self.best_sp:
            self.best_sp = PyFloorPlantProblem(self.n)
            self.fpp = self.best_sp

        sa_cost = (-self.obj) - (-self.best_obj) + 10e-8
        print(sa_cost, np.exp(sa_cost/min(-self.sa_temp, -1e-8)), np.random.rand())
        if sa_cost < 0 or np.exp(sa_cost/min(-self.sa_temp, -1e-8)) > np.random.rand():
            self.best_sp = self.fpp

        self.fpp = self.best_sp.copy()
        # self.fpp.shuffle_sp()

        self.best_obj = -self.fpp.get_current_sp_objective()
        self.obj = self.best_obj

        self.steps = 0

        self.observation = tuple([
            tuple(to_positions(self.fpp.x())),
            tuple(to_positions(self.fpp.y())),
        ])
        self.sa_temp = self.sa_temp*self.sa_alpha
        assert self.observation_space.contains(self.observation)

    def step(self, action: tuple[int, int, int]):
        assert self.action_space.contains(action)
        #print("Taking action: ", action)
        i, j, move = action

        self.steps += 1
        if move == 9 or self.steps > self.max_steps:
            obj_diff = self.obj - self.best_obj
            # if obj_diff < 0:
            #    obj_diff = 0
            return self.observation, 0, True, {}

        if i >= self.n or j >= self.n or i == j:
            return self.observation, 0, False, {}

        self.previous_obj = self.obj
        self.obj = -self.fpp.apply_sp_move(i, j, move)

        self.observation = tuple([
            tuple(to_positions(self.fpp.x())),
            tuple(to_positions(self.fpp.y())),
        ])
        assert self.observation_space.contains(self.observation)

        return self.observation, self.obj - self.previous_obj, False, {}

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



