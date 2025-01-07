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

import tensorflow as tf
from sklearn.cluster import SpectralClustering
import matplotlib.pyplot as plt

def save_floorplan_image(canvas_tensor, file_path="floorplan.png"):
    """
    Function to save the floorplan image to disk.

    Parameters:
    - canvas_tensor: The TensorFlow tensor representing the canvas (height, width, 3).
    - file_path: The path to save the image.
    """
    # Convert TensorFlow tensor to a NumPy array
    canvas_np = canvas_tensor.numpy()

    # Normalize to the range [0, 255] for image visualization
    canvas_np = np.clip(canvas_np * 255, 0, 255).astype(np.uint8)

    # Plot the image using Matplotlib
    plt.imshow(canvas_np)
    plt.axis('off')  # Turn off axes
    plt.savefig(file_path, bbox_inches='tight', pad_inches=0)  # Save image to disk
    plt.close()  # Close the plot to avoid display blocking

def generate_colors_from_connections(adj_matrix, num_components):
    """
    Generate RGB colors for components based on their connections.
    Connected components will have similar colors, and components
    with more connections will have brighter colors.

    Parameters:
    - adj_matrix: Adjacency matrix representing connections between components.
    - num_components: Number of components.

    Returns:
    - An array of RGB colors (num_components x 3).
    """
    # Perform clustering based on the adjacency matrix
    clustering = SpectralClustering(
        n_clusters=min(num_components, 10), affinity="precomputed", random_state=42
    )
    labels = clustering.fit_predict(adj_matrix)

    # Calculate connection strength for brightness interpolation
    connection_strength = np.sum(adj_matrix, axis=1)  # Sum of connections per node
    max_strength = np.max(connection_strength)
    min_strength = np.min(connection_strength)
    brightness_scale = (connection_strength - min_strength) / (max_strength - min_strength + 1e-6)  # Normalize to [0, 1]

    # Generate equidistant base colors for clusters
    base_colors = np.linspace(0, 1, min(num_components, 10))
    np.random.shuffle(base_colors)
    cluster_colors = np.array([base_colors[labels],
                               np.roll(base_colors, 1)[labels],
                               np.roll(base_colors, 2)[labels]]).T

    # Add randomization to base colors and adjust brightness
    colors = []
    for i, color in enumerate(cluster_colors):
        random_offset = np.random.uniform(-0.1, 0.1, size=3)  # Add small random variation
        brightened_color = color + random_offset  # Apply random offset
        brightened_color = np.clip(brightened_color, 0, 1)  # Ensure valid RGB values
        brightened_color *= 0.2 + 0.8 * brightness_scale[i]  # Adjust brightness based on connection strength
        brightened_color = np.clip(brightened_color, 0, 1)  # Clip again after brightness adjustment
        colors.append(brightened_color)

    return np.array(colors)


def draw_floorplan_with_colors(
    widths, heights, h_offsets, v_offsets, colors, canvas_size=(64, 64), steps=1, max_steps=1
):
    canvas_height, canvas_width = canvas_size
    # Convert inputs to tensors for TensorFlow operations
    widths = tf.convert_to_tensor(widths, dtype=tf.int32)
    heights = tf.convert_to_tensor(heights, dtype=tf.int32)
    h_offsets = tf.convert_to_tensor(h_offsets, dtype=tf.int32)
    v_offsets = tf.convert_to_tensor(v_offsets, dtype=tf.int32)
    colors = tf.convert_to_tensor(colors, dtype=tf.float32)

    canvas = tf.zeros((canvas_height, canvas_width, 3), dtype=tf.float32)

    # Create a grid representing all pixel coordinates
    y_grid, x_grid = tf.meshgrid(tf.range(canvas_height), tf.range(canvas_width), indexing='ij')

    # Broadcast dimensions for vectorized comparison
    y_grid = tf.expand_dims(y_grid, axis=-1)  # Shape: (canvas_height, canvas_width, 1)
    x_grid = tf.expand_dims(x_grid, axis=-1)  # Shape: (canvas_height, canvas_width, 1)

    # Calculate boundaries for each rectangle
    x_start = h_offsets
    x_end = h_offsets + widths
    y_start = v_offsets
    y_end = v_offsets + heights

    # Create masks for all rectangles
    x_in_rect = (x_grid >= x_start) & (x_grid < x_end)
    y_in_rect = (y_grid >= y_start) & (y_grid < y_end)
    in_rect = x_in_rect & y_in_rect  # Shape: (canvas_height, canvas_width, num_rectangles)

    # Find which rectangle each pixel belongs to
    rect_mask = tf.reduce_any(in_rect, axis=-1)  # Combined mask for all rectangles
    rect_indices = tf.argmax(tf.cast(in_rect, tf.int32), axis=-1)  # Index of the rectangle for each pixel

    # Apply colors to the pixels
    pixel_colors = tf.where(
        tf.expand_dims(rect_mask, axis=-1),  # Condition: pixel belongs to any rectangle
        tf.gather(colors, rect_indices),    # Colors for the corresponding rectangles
        canvas                              # Keep the original canvas color (black)
    )

    return pixel_colors


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
    max_steps: int = 50
    steps: int = 0

    def __init__(self, n: int):
        self.n = n
        self.max_offset = n
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
            spaces.Tuple([
                 spaces.Tuple([
                     spaces.Discrete(2)
                     for j in range(self.n)
                 ])
                for i in range(self.n)
            ]),
            # Y
            spaces.Tuple([
                spaces.Tuple([
                    spaces.Discrete(2)
                    for j in range(self.n)
                ])
                for i in range(self.n)
            ]),
            # offset widths
            spaces.Box(low=0, high=np.inf, shape=(n,)),
            # offset heights
            spaces.Box(low=0, high=np.inf, shape=(n,)),
            # weighted_connections
            spaces.Tuple([
                spaces.Box(low=0, high=np.inf, shape=(self.n-i-1,))
                for i in range(self.n-1)
            ]),
        ])
        self.reset()
        super().__init__()

    def flattened_observation(self) -> tuple[np.array, np.array]:
        xy = []
        for row in self.observation[0]:
            xy.extend(np.ndarray.flatten(np.array(list(row))))
        for row in self.observation[1]:
            xy.extend(np.ndarray.flatten(np.array(list(row))))

        reals = []
        reals.extend(np.ndarray.flatten(np.array(list(self.observation[2]))))
        reals.extend(np.ndarray.flatten(np.array(list(self.observation[3]))))
        for row in self.observation[4]:
            reals.extend(np.ndarray.flatten(np.array(list(row))))
        return np.array(xy), np.array(reals)

    def draw(self, fpp: PyFloorPlantProblem):
         return draw_floorplan_with_colors(
            self.widths,
            self.heights,
            fpp.offset_widths(),
            fpp.offset_heights(),
            self.colors,
            (int(self.max_offset), int(self.max_offset)),
            self.steps,
            self.max_steps
        )

    def reset(self):
        if not self.fpp:
            self.ini_fpp = PyFloorPlantProblem(self.n)
            self.fpp = self.ini_fpp.copy()
            self.max_offset = max(self.fpp.offset_widths() + self.fpp.offset_heights())
            for _ in range(100):
                self.fpp.shuffle_sp()
                self.max_offset = max([self.max_offset, *self.fpp.offset_widths(), *self.fpp.offset_heights()])
            self.max_offset = 4*(np.round(self.max_offset/4))
            print("Max offset:", self.max_offset)
            self.colors = generate_colors_from_connections(self.ini_fpp.connected_to(), self.n)
            self.widths = self.ini_fpp.widths()
            self.heights = self.ini_fpp.heights()
            save_floorplan_image(self.draw(self.ini_fpp), "visualizations/ini_fpp_visualization.png")
            self.best_fpp = self.ini_fpp.copy()
            self.best_obj = self.best_fpp.get_current_sp_objective()

            self.ini_obj = self.ini_fpp.get_current_sp_objective()
            self.fpp = self.ini_fpp.copy()

            self.sa_fpp = self.fpp.copy()
            print("Simulated Annealing...")
            self.sa_fpp.apply_simulated_annealing(100, 1.0-1e-6)
            print("Simulated Annealing result: ", self.sa_fpp.get_current_sp_objective())
            self.sa_fpp.visualize()
            save_floorplan_image(self.draw(self.sa_fpp), "visualizations/sa_fpp_visualization.png")

            self.rand_ini_fpp = self.fpp.copy()
            self.rand_best_fpp = self.fpp.copy()


        self.fpp = self.ini_fpp.copy()
        self.min_fpp = self.ini_fpp.copy()
        self.rand_fpp = self.rand_ini_fpp.copy()
        self.min_rand_fpp = self.rand_ini_fpp.copy()

        #self.fpp.shuffle_sp()
        #self.rand_fpp.shuffle_sp()
        self.previous_obj = self.fpp.get_current_sp_objective()

        self.steps = 0
        self.observation = None

    def step(self, action: tuple[int, int, int], just_step: bool = False):
        assert self.action_space.contains(action)
        i, j, move = action
        self.steps += 1
        if i >= self.n or j >= self.n or i == j:
            print("Stop!")
            print(i, j)
            return self.observation, -1, True, {}

        done = move == 9 or self.steps >= self.max_steps
        previous_obj = self.fpp.get_current_sp_objective()
        if move < 9:
            self.fpp.apply_sp_move(i, j, move)

            first_choice, second_choice = np.random.choice(self.n, 2, replace=False)
            self.rand_fpp.apply_sp_move(first_choice, second_choice, random.randint(0, 2))

#         self.fpp.apply_simulated_annealing(0.165, 1.0-1e-3)
#         self.rand_fpp.apply_simulated_annealing(0.165, 1.0-1e-3)

        obj = self.fpp.get_current_sp_objective()
        rand_obj = self.rand_fpp.get_current_sp_objective()
        if obj < self.best_obj:
            self.best_fpp = self.fpp.copy()
            self.best_obj = obj
        if rand_obj < self.rand_best_fpp.get_current_sp_objective():
            self.rand_best_fpp = self.rand_fpp.copy()

        return self.observation, (previous_obj-obj)/self.ini_obj, done, {}
        #return self.observation, max((self.ini_obj-obj)/self.ini_obj, 0) if done else 0, done, {}

    def render(self):
        self.fpp.visualize()