from collections import deque
import numpy as np
import tensorflow as tf
from tensorflow import keras, GradientTape
import random
import os
from floorplant_gym_env import FloorPlantEnv
from tensorflow.keras.layers import MultiHeadAttention, LayerNormalization, Dense, Dropout, Lambda

print("Num GPUs Available: ",
    len(tf.config.experimental.list_physical_devices('GPU')))
# # Configure TensorFlow to use mixed precision for faster computation
# tf.keras.mixed_precision.set_global_policy('mixed_float16')

# Set threading options
tf.config.threading.set_inter_op_parallelism_threads(8)  # Adjust based on your CPU cores
tf.config.threading.set_intra_op_parallelism_threads(8)


# Initialize environment
env = FloorPlantEnv(30)
n_moves = 9

# Define model parameters
hidden_nodes = 2**8

# Actor network with attention
actor_input_layer = keras.layers.Input((env.n + env.n,))  # Input shape
# actor_embedding = keras.layers.Embedding(input_dim=env.n, output_dim=int(np.ceil(np.log2(env.n))))(actor_input_layer)
# actor_flatten = keras.layers.Flatten()(actor_embedding)

# # Reshape to (batch_size, seq_len, feature_dim) for attention
# reshaped_input = Lambda(lambda x: tf.expand_dims(x, axis=1))(actor_flatten)
#
# # Apply multi-head attention
# attention_output = MultiHeadAttention(num_heads=2, key_dim=32)(
#     query=reshaped_input, key=reshaped_input, value=reshaped_input
# )
#
# # Normalize and add dropout
# attention_output = LayerNormalization()(attention_output)
#
# # Flatten for dense layers
# attention_flattened = tf.keras.layers.Flatten()(attention_output)

# Dense layers
actor_hidden_layer = tf.keras.layers.LeakyReLU(negative_slope=0.1)(Dense(hidden_nodes)(actor_input_layer))
actor_hidden_layer = tf.keras.layers.LeakyReLU(negative_slope=0.1)(Dense(hidden_nodes)(actor_hidden_layer))
actor_hidden_layer = tf.keras.layers.LeakyReLU(negative_slope=0.1)(Dense(hidden_nodes)(actor_hidden_layer))
actor_hidden_layer = tf.keras.layers.LeakyReLU(negative_slope=0.1)(Dense(hidden_nodes)(actor_hidden_layer))
# actor_hidden_layer = Dense(hidden_nodes, activation="tanh")(actor_hidden_layer)

# Outputs
wfa = Dense(env.n, activation="softmax")(actor_hidden_layer)
wsa = Dense(env.n, activation="softmax")(actor_hidden_layer)
wma = Dense(n_moves, activation="softmax")(actor_hidden_layer)
critic = keras.layers.Dense(1)(actor_hidden_layer)
# Compile actor model
model = keras.Model(inputs=actor_input_layer, outputs=[wfa, wsa, wma, critic])

# Critic network
# critic_input_layer = keras.layers.Input((env.n + env.n,))
# critic_hidden_layer = keras.layers.Embedding(env.n, int(np.ceil(np.log2(env.n))))(critic_input_layer)
# critic_hidden_layer = keras.layers.Flatten()(critic_hidden_layer)
# critic_hidden_layer = keras.layers.LeakyReLU(negative_slope=0.1)(keras.layers.Dense(hidden_nodes)(critic_hidden_layer))
# critic_hidden_layer = keras.layers.LeakyReLU(negative_slope=0.1)(keras.layers.Dense(hidden_nodes)(critic_hidden_layer))
# critic_output = keras.layers.Dense(1)(critic_hidden_layer)

# critic = keras.Model(inputs=critic_input_layer, outputs=critic_output)
optimizer = keras.optimizers.Adam(learning_rate=1e-3)

# Training parameters
replay_buffer = deque(maxlen=10000)
batch_size = 64
gamma = 0.99
entropy_coefficient = 0*1e-7
num_episodes = 10000
eps = np.finfo(np.float32).eps.item()

# Training loop
running_reward = 0
for episode in range(num_episodes):
    if episode % 10 == 0:
#             search_epsilon = 0.5
        print("Simulated Annealing solution:", env.sa_fpp.get_current_sp_objective())
        env.sa_fpp.visualize()
        env.best_fpp.visualize()
    env.reset()
    state = np.ravel(env.flattened_observation()[0])  # Flatten state
    episode_reward = 0
    done = False

    while not done:
        inp = tf.expand_dims(tf.convert_to_tensor(state, dtype=tf.float32), 0)
        wfp, wsp, wmp, critic_value = model(inp)

        # Action sampling
        wfp_dist = wfp.numpy()[0]
        wsp_dist = wsp.numpy()[0]
        wmp_dist = wmp.numpy()[0]

        wfp_dist /= sum(wfp_dist)
        first_choice = np.random.choice(env.n, p=wfp_dist)

        wsp_dist[first_choice] = 0
        wsp_dist /= sum(wsp_dist)
        second_choice = np.random.choice(env.n, p=wsp_dist)

        wmp_dist /= sum(wmp_dist)
        move = np.random.choice(n_moves, p=wmp_dist)

        action = (first_choice, second_choice, move)
        _, reward, done, _ = env.step(action)
        next_state = np.ravel(env.flattened_observation()[0])

        # Store transition in replay buffer
        replay_buffer.append((state, action, reward, next_state, done))
        state = next_state
        episode_reward += reward

        # Initialize loss trackers
        total_critic_loss = 0.0
        total_actor_loss = 0.0
        update_steps = 0  # Count the number of updates per episode

        # Perform batch updates
        if len(replay_buffer) >= batch_size:
            batch = random.sample(replay_buffer, batch_size)
            states, actions, rewards, next_states, dones = zip(*batch)

            # Convert batch to tensors
            states = tf.convert_to_tensor(np.array(states), dtype=tf.float32)
            next_states = tf.convert_to_tensor(np.array(next_states), dtype=tf.float32)
            rewards = tf.convert_to_tensor(rewards, dtype=tf.float32)
            dones = tf.convert_to_tensor(dones, dtype=tf.float32)

            # Update
            with GradientTape() as tape:
                critic_values = tf.squeeze(model(states)[3], axis=-1)
                critic_next_values = tf.squeeze(model(next_states)[3], axis=-1)
                target_values = rewards + gamma * critic_next_values * (1.0 - dones)
                advantages = target_values - critic_values
                critic_loss = tf.reduce_mean(advantages ** 2)

            # Backpropagate loss
            grads = tape.gradient(
                (tf.constant(0), tf.constant(0), tf.constant(0), critic_loss),
                model.trainable_variables
            )
            optimizer.apply_gradients(zip(grads, model.trainable_variables))

            with GradientTape() as tape:
                # Predict actions for the batch
                predicted_wfp, predicted_wsp, predicted_wmp, _ = model(states)

                # Create masks for selected actions
                chosen_wfp = tf.one_hot([a[0] for a in actions], env.n)
                chosen_wsp = tf.one_hot([a[1] for a in actions], env.n)
                chosen_wmp = tf.one_hot([a[2] for a in actions], n_moves)

                # Compute log probabilities for selected actions
                log_probs_wfp = tf.reduce_sum(chosen_wfp * tf.math.log(predicted_wfp + eps), axis=1)
                log_probs_wsp = tf.reduce_sum(chosen_wsp * tf.math.log(predicted_wsp + eps), axis=1)
                log_probs_wmp = tf.reduce_sum(chosen_wmp * tf.math.log(predicted_wmp + eps), axis=1)

                # Compute actor loss weighted by advantages
                actor_loss_wfp = -tf.reduce_mean(log_probs_wfp * advantages)
                actor_loss_wsp = -tf.reduce_mean(log_probs_wsp * advantages)
                actor_loss_wmp = -tf.reduce_mean(log_probs_wmp * advantages)

                # Add entropy regularization
                entropy_loss_wfp = entropy_coefficient * tf.reduce_sum(predicted_wfp * tf.math.log(predicted_wfp + eps))
                entropy_loss_wsp = entropy_coefficient * tf.reduce_sum(predicted_wsp * tf.math.log(predicted_wsp + eps))
                entropy_loss_wmp = entropy_coefficient * tf.reduce_sum(predicted_wmp * tf.math.log(predicted_wmp + eps))

                # Iterate through batch
#                 for idx, (state, action) in enumerate(zip(states, actions)):
#                     first_choice, second_choice, move = action
#                     inp = tf.expand_dims(state, 0)
#                     wfp, wsp, wmp, _ = model(inp)
#
#                     # Log probabilities for chosen actions
#                     log_wfp = tf.math.log(wfp[0, first_choice] + eps)
#                     log_wsp = tf.math.log(wsp[0, second_choice] + eps)
#                     log_wmp = tf.math.log(wmp[0, move] + eps)
#
#                     # Compute individual losses weighted by advantage
#                     actor_loss_wfp += -log_wfp * advantages[idx]
#                     actor_loss_wsp += -log_wsp * advantages[idx]
#                     actor_loss_wmp += -log_wmp * advantages[idx]
#
#                 # Add entropy regularization (to encourage exploration) for each head
#                 entropy_loss_wfp = entropy_coefficient * tf.reduce_sum(wfp * tf.math.log(wfp + eps))
#                 entropy_loss_wsp = entropy_coefficient * tf.reduce_sum(wsp * tf.math.log(wsp + eps))
#                 entropy_loss_wmp = entropy_coefficient * tf.reduce_sum(wmp * tf.math.log(wmp + eps))
#
                # Combine all losses
                actor_loss = (
                    actor_loss_wfp + entropy_loss_wfp,
                    actor_loss_wsp + entropy_loss_wsp,
                    actor_loss_wmp + entropy_loss_wmp,
                )

            # Backpropagate loss
            grads = tape.gradient((*actor_loss, tf.constant(0)), model.trainable_variables)
            optimizer.apply_gradients(zip(grads, model.trainable_variables))

            # Accumulate losses for logging
            total_actor_loss += tf.reduce_sum(actor_loss)
            total_critic_loss += critic_loss.numpy()
            update_steps += 1

    # Update running reward
    running_reward = 0.05 * episode_reward + (1 - 0.05) * running_reward

    # Log details
    avg_critic_loss = total_critic_loss / max(update_steps, 1)
    avg_actor_loss = total_actor_loss / max(update_steps, 1)

    template = (
        "Rand best obj: {:.2f}, "
        "Best obj {:.2f}, "
        "Running reward: {:.2f}, "
        "Critic loss: {:.4f}, "
        "Actor loss: {:.4f}, "
        "Episode final state: {:.2f}"
    )
    print(template.format(
        env.rand_best_fpp.get_current_sp_objective(),
        env.best_obj,
        running_reward,
        avg_critic_loss,
        avg_actor_loss,
        env.fpp.get_current_sp_objective(),
    ))

    # Reset loss trackers after logging
    total_critic_loss = 0.0
    total_actor_loss = 0.0
    update_steps = 0

env.close()
