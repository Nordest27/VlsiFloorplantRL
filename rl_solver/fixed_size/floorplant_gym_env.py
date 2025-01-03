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

class FloorPlantEnv(gym.Env):

    fpp: PyFloorPlantProblem = None
    n: int
    observation: tuple
    initial_obj: int
    previous_obj: int
    obj: int
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
            # widths
            spaces.Box(low=1, high=np.inf, shape=(n,)),
            # heights
            spaces.Box(low=1, high=np.inf, shape=(n,)),
            # offset widths
            spaces.Box(low=0, high=np.inf, shape=(n,)),
            # offset heights
            spaces.Box(low=0, high=np.inf, shape=(n,)),
            # connected to
            spaces.Tuple([
                spaces.Tuple([
                    spaces.Discrete(2)
                for _ in range(self.n)])
            for _ in range(self.n)])
        ])
        self.seed()
        self.reset()
        super().__init__()

    def to_one_hot_encoding(self, t: tuple[int]) -> list[list[int]]:
        return [[int(v==i) for i in range(self.n)] for v in  t]

    def flattened_observation(self) -> np.array:
        observation = []

        for ob in self.observation:
            try:
                observation.extend(np.ndarray.flatten(np.array(list(ob))))
            except:
                try:
                    for ob1 in ob:
                        observation.extend(np.ndarray.flatten(np.array(list(ob1))))
                except:
                    for ob2 in ob:
                        observation.extend(np.ndarray.flatten(np.array(list(ob2))))
        return np.array(observation)

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def reset(self):
        self.fpp = PyFloorPlantProblem(self.n)

        # self.fpp.shuffle_sp()

        self.initial_obj = -self.fpp.get_current_sp_objective()
        self.obj = self.initial_obj

        self.steps = 0

        connected_to = tuple([tuple([int(v) for v in row]) for row in self.fpp.connected_to()])

        self.observation = tuple([
            tuple(self.fpp.x()),
            tuple(self.fpp.y()),
            to_numpy_array(self.fpp.widths()),
            to_numpy_array(self.fpp.heights()),
            to_numpy_array(self.fpp.offset_widths()),
            to_numpy_array(self.fpp.offset_heights()),
            tuple(connected_to)
        ])

        assert self.observation_space.contains(self.observation)

    def step(self, action: tuple[int, int, int]):
        assert self.action_space.contains(action)
        #print("Taking action: ", action)
        i, j, move = action

        self.steps += 1
        if move == 9 or self.steps > self.max_steps:
            obj_diff = self.obj - self.initial_obj
            if obj_diff < 0:
                obj_diff = -1
            return self.observation, obj_diff, True, {}

        if i >= self.n or j >= self.n or i == j:
            return self.observation, -1, False, {}

        self.previous_obj = self.obj
        self.obj = -self.fpp.apply_sp_move(i, j, move)
        self.observation = tuple([
            tuple(self.fpp.x()),
            tuple(self.fpp.y()),
            self.observation[2],
            self.observation[3],
            to_numpy_array(self.fpp.offset_widths()),
            to_numpy_array(self.fpp.offset_heights()),
            self.observation[6],
        ])
        assert self.observation_space.contains(self.observation)

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



