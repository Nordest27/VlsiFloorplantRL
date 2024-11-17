# using example https://keras.io/examples/rl/actor_critic_cartpole/
import numpy as np
import os
#os.environ["KERAS_BACKEND"] = "tensorflow"
from keras import backend
import tensorflow as tf
from tensorflow.python.keras.backend import set_session

# adjust values to your needs
config = tf.compat.v1.ConfigProto( device_count = {'GPU': 1 , 'CPU': 4} )
sess = tf.compat.v1.Session(config=config)
set_session(sess)


import gym
from floorplant_gym_env import FloorPlantEnv
import numpy as np


# Create the CartPole Environment
env = FloorPlantEnv(5)
n_moves = 3
output_moves = env.n*(env.n-1)*n_moves//2
epsilon = 0.0

# Define the actor and critic networks
input_layer = tf.keras.layers.Input((env.n + env.n,))

hidden_layer = tf.keras.layers.Embedding(env.n, 4)(input_layer)
hidden_layer = tf.keras.layers.Flatten()(hidden_layer)
hidden_layer = tf.keras.layers.LeakyReLU(negative_slope=0.1)(tf.keras.layers.Dense(512)(hidden_layer))
hidden_layer = tf.keras.layers.LeakyReLU(negative_slope=0.1)(tf.keras.layers.Dense(512)(hidden_layer))
actor = tf.keras.layers.Dense(output_moves, activation="softmax")(hidden_layer)
critic = tf.keras.layers.Dense(1)(hidden_layer)
model = tf.keras.Model(inputs=input_layer, outputs=[actor, critic])


# Define optimizer and loss functions
optimizer = tf.keras.optimizers.Adam(learning_rate=1e-3)


def ucb_score(parent, child):
    prior_score = child.prior * np.sqrt(parent.visit_count) / (child.visit_count + 1)
    if child.visit_count > 0:
        value_score = child.value
    else:
        value_score = 0

    return value_score + prior_score

class McNode:
    def __init__(self, prior, x, y):
        self.prior = prior      # The prior probability of selecting this state from its parent

        self.visit_count = 0    # Number of times this state was visited during MCTS. "Good" are visited more then "bad" states.
        self.value = 0      # The total value of this state from all visits
        self.x = x
        self.y = y
        self.children = {}

    def expanded(self):
        return len(self.children) > 0

    def expand(self, action_probs):
        """
        We expand a node and keep track of the prior policy probability given by neural network
        """
        env.fpp.set_sp(self.x, self.y)
        self.value = env.current_obj - env.fpp.get_current_sp_objective()
        for move, prob in enumerate(action_probs):
            env.fpp.set_sp(self.x, self.y)
            count = 0
            first_choice, second_choice, move_choice = (0, 1, 9)
            for i in range(env.n-1):
                for j in range(env.n-i-1):
                    for m in range(n_moves):
                        if count == action:
                            first_choice, second_choice, move_choice = i, i+j+1, m
                        count += 1
            env.fpp.apply_sp_move((first_choice, second_choice, move_choice))
            self.children[move] = McNode(prob, env.fpp.x(), env.fpp.y())

    def select_action(self, temperature):
        """
        Select action according to the visit count distribution and the temperature.
        """
        visit_counts = np.array([child.visit_count for child in self.children.values()])
        if temperature == 0:
            action = np.argmax(visit_counts)
        elif temperature == float("inf"):
            action = np.random.choice(output_moves)
        else:
            # See paper appendix Data Generation
            visit_count_distribution = visit_counts ** (1 / temperature)
            visit_count_distribution = visit_count_distribution / sum(visit_count_distribution)
            action = np.random.choice(output_moves, p=visit_count_distribution)

        return action

    def select_child(self):
        """
        Select the child with the highest UCB score.
        """
        best_score = -np.inf
        best_child = None

        for action, child in self.children.items():
            score = ucb_score(self, child)
            if score > best_score:
                best_score = score
                best_child = child

        return best_child

    def backpropagate(self, search_path, value):
        """
        At the end of a simulation, we propagate the evaluation all the way up the tree
        to the root.
        """
        for node in reversed(search_path):
            node.value_sum += value
            node.visit_count += 1

def mc_simulation(self, x, y, num_simulations):
    root = McNode(0, x, y)

    # EXPAND root
    action_probs, pred_value = model(tf.expand_dims(env.flattened_xy_positions(), 0))
    root.expand(action_probs)

    for _ in range(num_simulations):
        node = root
        search_path = [node]

        # SELECT
        while node.expanded():
            node = node.select_child()
            search_path.append(node)

        env.fpp.set_sp(node.x, node.y)

        # Expand
        action_probs, value = model.predict(next_state)
        valid_moves = self.game.get_valid_moves(next_state)
        action_probs = action_probs * valid_moves  # mask invalid moves
        action_probs /= np.sum(action_probs)
        node.expand(next_state, parent.to_play * -1, action_probs)

        self.backpropagate(search_path, value)

    return root


n_batches = 2
n_sequence = 3
for it in range(100):
    with tf.GradientTape() as tape:
        loss_value = 0
        for batch in range(n_batches):
            env.reset()
            for i_seq in range(n_sequence):

                action_probs = model(tf.expand_dims(env.flattened_xy_positions(), 0))

                # Calculate loss directly with tensor operations
                # loss_value = tf.square(monte_carlo_result - action_values)
                loss_value -= tf.keras.ops.log(action_probs[0, mc_action])/(n_batches*n_sequence)

                count = 0
                first_choice, second_choice, move = (0, 1, 9)
                for i in range(env.n-1):
                    for j in range(env.n-i-1):
                        for m in range(n_moves):
                            if count == action:
                                first_choice, second_choice, move = i, i+j+1, m
                            count += 1
                env.think_step((first_choice, second_choice, move))

        print(f"{it} | Loss: {loss_value}")
        grads = tape.gradient(loss_value, model.trainable_variables)
        optimizer.apply_gradients(zip(grads, model.trainable_variables))

env.reset()
env.fpp.visualize()
for i in range(n_sequence):
    action_probs = model(tf.expand_dims(env.flattened_xy_positions(), 0))
    # action_probs = tf.math.exp(action_values) / tf.reduce_sum(tf.math.exp(action_values))
    action = np.argmax(action_probs[0].numpy())

    count = 0
    first_choice, second_choice, move = (0, 1, 9)
    for i in range(env.n-1):
        for j in range(env.n-i-1):
            for m in range(n_moves):
                if count == action:
                    first_choice, second_choice, move = i, i+j+1, m
                count += 1

    print(first_choice, second_choice, move)
    env.think_step((first_choice, second_choice, move))
    env.fpp.visualize()
