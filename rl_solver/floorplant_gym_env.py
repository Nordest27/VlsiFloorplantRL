# Example from:
# https://stackoverflow.com/questions/44469266/how-to-implement-custom-environment-in-keras-rl-openai-gym
# https://github.com/openai/gym/blob/pr/429/gym/envs/toy_text/hotter_colder.py
import string

import gym
from gym import spaces
from gym.utils import seeding
import numpy as np

from vlsi_floorplant import PyFloorPlantProblem

class FloorPlantEnv(gym.Env):

    def __init__(self, n: int):
        self.action_space = spaces.Tuple(
            spaces=[
                # Which first
                spaces.Text(
                    min_length=1,
                    max_length=3,
                    charset=string.digits
                ),
                # Which second
                spaces.Text(
                    min_length=1,
                    max_length=3,
                    charset=string.digits
                ),
                # Type of move
                spaces.Discrete(9)
            ]
        )
        self.fpp = PyFloorPlantProblem(n)
        self.observation_space = spaces.Tuple(
            spaces=[
                # X
                spaces.Sequence(spaces.Text(1, charset=string.digits)),
                # Y
                spaces.Sequence(spaces.Text(1, charset=string.digits)),
                # widths
                spaces.Sequence(spaces.Box(low=np.array([1]), high=np.array([100]))),
                # heights
                spaces.Sequence(spaces.Box(low=np.array([1]), high=np.array([100]))),
                # connected to
                spaces.Sequence(spaces.Sequence(spaces.Discrete(2)))
            ]
        )
        self._seed()
        self._reset()

    def _seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def _step(self, action):
        assert self.action_space.contains(action)

    def _reset(self):
        pass

    def _render(self, mode='human'):
        self.fpp.visualize()

if __name__ == "__main__":
    n = 10
    fpe = FloorPlantEnv(n)
    print(fpe.action_space.sample())
    print(fpe.observation_space.sample())
    print(fpe.fpp.x)
    print(fpe.fpp.y)
    print(fpe.fpp.widths)
    print(fpe.fpp.heights)
    print(fpe.fpp.connected_to)
    print(fpe.fpp.visualize())



