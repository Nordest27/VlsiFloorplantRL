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

model = tf.keras.Model(inputs=input_layer, outputs=actor)

monte_carlo_dists = {}
# Define optimizer and loss functions
optimizer = tf.keras.optimizers.Adam(learning_rate=1e-3)
n_batches = 10
n_sequence = 5
for it in range(100):
    with tf.GradientTape() as tape:
        loss_value = 0
        for batch in range(n_batches):
            env.reset()
            for i_seq in range(n_sequence):
                inp = env.flattened_xy_positions()
                if str(inp) not in monte_carlo_dists:
                    print("Executing monte carlo exploration for:", str(inp), "at i_seq ", i_seq)
                    monte_carlo_dists[str(inp)] = env.get_rand_monte_carlo_dist(100000, 1)
                monte_carlo_result = monte_carlo_dists[str(inp)]
                monte_carlo_result = tf.convert_to_tensor(monte_carlo_result, dtype=tf.float32)
                monte_carlo_result = tf.math.exp(monte_carlo_result) / tf.reduce_sum(tf.math.exp(monte_carlo_result))
                monte_carlo_result = tf.reshape(monte_carlo_result, (output_moves,))
                mc_action = np.argmax(monte_carlo_result)
                if monte_carlo_result[mc_action] <= 0:
                    print("Monte carlo found nothing")
                    continue

                action_probs = model(tf.expand_dims(inp, 0))
                action = np.random.choice(output_moves, p=action_probs.numpy()[0])  # Best action

                # Calculate loss directly with tensor operations
                # loss_value = tf.square(monte_carlo_result - action_values)
                loss_value -= tf.keras.ops.log(action_probs[0, mc_action])/(n_batches*n_sequence)

                count = 0
                first_choice, second_choice, move = (0, 1, 9)
                for i in range(env.n-1):
                    for j in range(env.n-i-1):
                        for m in range(n_moves):
                            if count == mc_action:
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
