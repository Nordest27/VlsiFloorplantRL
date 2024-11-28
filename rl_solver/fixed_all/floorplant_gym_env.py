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
        widths, heights, h_offsets, v_offsets, colors, canvas_size=(64, 64)
):
    # Calculate the total layout dimensions
    total_width = max(h_offsets[i] + widths[i] for i in range(len(widths)))
    total_height = max(v_offsets[i] + heights[i] for i in range(len(heights)))

    # Compute scale factors
    scale_x = canvas_size[1] / total_width
    scale_y = canvas_size[0] / total_height
    scale_factor = min(scale_x, scale_y)

    # Create blank canvas
    canvas_height, canvas_width = canvas_size
    canvas = tf.zeros((canvas_height, canvas_width, 3), dtype=tf.float32)

    # Sort components by size
    component_indices = sorted(range(len(widths)), key=lambda i: widths[i] * heights[i])

    # Draw each component
    for i in component_indices:
        w, h, x_offset, y_offset = widths[i], heights[i], h_offsets[i], v_offsets[i]
        scaled_w = max(int(w * scale_factor), 1)
        scaled_h = max(int(h * scale_factor), 1)
        scaled_x_offset = int(x_offset * scale_factor)
        scaled_y_offset = int(y_offset * scale_factor)

        # Ensure valid coordinates
        x_start, x_end = max(0, scaled_x_offset), min(canvas_width, scaled_x_offset + scaled_w)
        y_start, y_end = max(0, scaled_y_offset), min(canvas_height, scaled_y_offset + scaled_h)

        if x_start < x_end and y_start < y_end:
            y_coords, x_coords = tf.meshgrid(
                tf.range(y_start, y_end), tf.range(x_start, x_end), indexing="ij"
            )
            flat_indices = tf.reshape(tf.stack([y_coords, x_coords], axis=-1), (-1, 2))
            flat_colors = tf.repeat(tf.convert_to_tensor(colors[i])[None, :], flat_indices.shape[0], axis=0)

            # Align data types
            flat_indices = tf.cast(flat_indices, tf.int32)
            flat_colors = tf.cast(flat_colors, canvas.dtype)

            # Update canvas
            canvas = tf.tensor_scatter_nd_update(canvas, flat_indices, flat_colors)

    return canvas



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
#         print(np.array(xy))
#         print(np.array(reals))
        return np.array(xy), np.array(reals)

    def draw(self, fpp: PyFloorPlantProblem):
         return draw_floorplan_with_colors(
            self.widths,
            self.heights,
            fpp.offset_widths(),
            fpp.offset_heights(),
            self.colors,
            (max(self.n, 32), max(self.n, 32))
        )

    def reset(self):
        if not self.fpp:
            self.ini_fpp = PyFloorPlantProblem(self.n)
            self.colors = generate_colors_from_connections(self.ini_fpp.connected_to(), self.n)
            self.widths = self.ini_fpp.widths()
            self.heights = self.ini_fpp.heights()
            save_floorplan_image(self.draw(self.ini_fpp), "visualizations/ini_fpp_visualization.png")
            self.best_fpp = self.ini_fpp.copy()
            self.best_obj = self.best_fpp.get_current_sp_objective()
            self.ini_obj = self.best_obj
            self.fpp = self.best_fpp.copy()

            self.sa_fpp = self.fpp.copy()
            print("Simulated Annealing...")
            self.sa_fpp.apply_simulated_annealing(100, 1.0-1e-2)
            print("Simulated Annealing result: ", self.sa_fpp.get_current_sp_objective())
            self.sa_fpp.visualize()
            save_floorplan_image(self.draw(self.sa_fpp), "visualizations/sa_fpp_visualization.png")

            self.rand_ini_fpp = self.fpp.copy()
            self.rand_best_fpp = self.fpp.copy()


        self.fpp = self.ini_fpp.copy()
        self.rand_fpp = self.rand_ini_fpp.copy()

        #self.fpp.shuffle_sp()
        #self.rand_fpp.shuffle_sp()
        self.previous_obj = self.fpp.get_current_sp_objective()

        self.steps = 0
        self.observation = None
        """
        self.observation = tuple([
            tuple(to_one_hot_encoding(self.fpp.x())),
            tuple(to_one_hot_encoding(self.fpp.y())),
            tuple(self.fpp.offset_widths()),
            tuple(self.fpp.offset_heights()),
            tuple(tuple(r) for r in self.fpp.weighted_connections())
        ])
        assert self.observation_space.contains(self.observation)
        """

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

        aux_fpp = self.fpp.copy()
        aux_rand_fpp = self.rand_fpp.copy()
        if not just_step:
            pass
            #aux_fpp.apply_simulated_annealing(0.101, 1.0-1e-3)
            #aux_rand_fpp.apply_simulated_annealing(0.101, 1.0-1e-3)

        obj = aux_fpp.get_current_sp_objective()
        rand_obj = aux_rand_fpp.get_current_sp_objective()
        if obj < self.best_obj:
            self.best_fpp = aux_fpp.copy()
            self.best_obj = obj
        if rand_obj < self.rand_best_fpp.get_current_sp_objective():
            self.rand_best_fpp = aux_rand_fpp.copy()
        """
        self.observation = tuple([
            tuple(to_one_hot_encoding(self.fpp.x())),
            tuple(to_one_hot_encoding(self.fpp.y())),
            tuple(self.fpp.offset_widths()),
            tuple(self.fpp.offset_heights()),
            tuple(tuple(r) for r in self.fpp.weighted_connections())
        ])
        assert self.observation_space.contains(self.observation)
        """
        #return self.observation, (previous_obj-obj)/self.ini_obj, move == 9 or self.steps > self.max_steps, {}
        return self.observation, (previous_obj-obj)/self.ini_obj, move == 9 or self.steps > self.max_steps, {}

    def render(self):
        self.fpp.visualize()