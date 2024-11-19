from collections import deque
import numpy as np
import tensorflow as tf
from tensorflow import keras, GradientTape
import random
from floorplant_gym_env import FloorPlantEnv

# Initialize environment
env = FloorPlantEnv(5)
n_moves = 9

# Define model parameters
hidden_nodes = 2**5

# Actor network
actor_input_layer = keras.layers.Input((env.n + env.n,))
actor_hidden_layer = keras.layers.Embedding(env.n, int(np.ceil(np.log2(env.n))))(actor_input_layer)
actor_hidden_layer = keras.layers.Flatten()(actor_hidden_layer)
actor_hidden_layer = keras.layers.LeakyReLU(negative_slope=0.1)(keras.layers.Dense(hidden_nodes)(actor_hidden_layer))
actor_hidden_layer = keras.layers.LeakyReLU(negative_slope=0.1)(keras.layers.Dense(hidden_nodes)(actor_hidden_layer))
actor_hidden_layer = keras.layers.Dense(hidden_nodes, activation="tanh")(actor_hidden_layer)

wfa = keras.layers.Dense(env.n, activation="softmax")(actor_hidden_layer)
wsa = keras.layers.Dense(env.n, activation="softmax")(actor_hidden_layer)
wma = keras.layers.Dense(n_moves, activation="softmax")(actor_hidden_layer)

actor = keras.Model(inputs=actor_input_layer, outputs=[wfa, wsa, wma])

# Critic network
critic_input_layer = keras.layers.Input((env.n + env.n,))
critic_hidden_layer = keras.layers.Embedding(env.n, int(np.ceil(np.log2(env.n))))(critic_input_layer)
critic_hidden_layer = keras.layers.Flatten()(critic_hidden_layer)
critic_hidden_layer = keras.layers.LeakyReLU(negative_slope=0.1)(keras.layers.Dense(hidden_nodes)(critic_hidden_layer))
critic_hidden_layer = keras.layers.LeakyReLU(negative_slope=0.1)(keras.layers.Dense(hidden_nodes)(critic_hidden_layer))
critic_output = keras.layers.Dense(1)(critic_hidden_layer)

critic = keras.Model(inputs=critic_input_layer, outputs=critic_output)

# Training parameters
search_epsilon = 0.1
replay_buffer = deque(maxlen=10000)
batch_size = 5
gamma = 0.99
entropy_coefficient = 0.01
regularization_strength = 1e-4
initial_actor_lr = 1e-2
initial_critic_lr = 1e-2
min_actor_lr = 1e-3
min_critic_lr = 1e-3
lr_decay_rate = 0.995
actor_optimizer = keras.optimizers.Adam(learning_rate=initial_actor_lr)
critic_optimizer = keras.optimizers.Adam(learning_rate=initial_critic_lr)
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
    state = np.ravel(env.flattened_observation())  # Flatten state
    episode_reward = 0
    done = False

    while not done:
        inp = tf.expand_dims(tf.convert_to_tensor(state, dtype=tf.float32), 0)
        critic_value = critic(inp)
        wfp, wsp, wmp = actor(inp)

        # Action sampling
        if random.random() < search_epsilon:
            first_choice, second_choice = np.random.choice(env.n, 2, replace=False)
            move = np.random.choice(n_moves)
        else:
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
        next_state, reward, done, _ = env.step(action)
        next_state = np.ravel(next_state)  # Flatten next state

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

            # Critic update
            with GradientTape() as tape:
                critic_values = tf.squeeze(critic(states), axis=-1)
                critic_next_values = tf.squeeze(critic(next_states), axis=-1)
                target_values = rewards + gamma * critic_next_values * (1.0 - dones)
                critic_loss = tf.reduce_mean((target_values - critic_values) ** 2)
            critic_grads = tape.gradient(critic_loss, critic.trainable_variables)
            critic_optimizer.apply_gradients(zip(critic_grads, critic.trainable_variables))

            # Actor update
            with GradientTape() as tape:
                advantages = target_values - critic_values

                # Initialize separate losses for each softmax head
                actor_loss_wfp = 0
                actor_loss_wsp = 0
                actor_loss_wmp = 0

                # Iterate through batch
                for idx, (state, action) in enumerate(zip(states, actions)):
                    first_choice, second_choice, move = action
                    inp = tf.expand_dims(state, 0)
                    wfp, wsp, wmp = actor(inp)

                    # Log probabilities for chosen actions
                    log_wfp = tf.math.log(wfp[0, first_choice] + eps)
                    log_wsp = tf.math.log(wsp[0, second_choice] + eps)
                    log_wmp = tf.math.log(wmp[0, move] + eps)

                    # Compute individual losses weighted by advantage
                    actor_loss_wfp += -log_wfp * advantages[idx]
                    actor_loss_wsp += -log_wsp * advantages[idx]
                    actor_loss_wmp += -log_wmp * advantages[idx]

                # Add entropy regularization (to encourage exploration) for each head
                entropy_loss_wfp = -entropy_coefficient * tf.reduce_sum(wfp * tf.math.log(wfp + eps))
                entropy_loss_wsp = -entropy_coefficient * tf.reduce_sum(wsp * tf.math.log(wsp + eps))
                entropy_loss_wmp = -entropy_coefficient * tf.reduce_sum(wmp * tf.math.log(wmp + eps))

                # Combine all losses
                actor_loss = (
                    actor_loss_wfp + entropy_loss_wfp,
                    actor_loss_wsp + entropy_loss_wsp,
                    actor_loss_wmp + entropy_loss_wmp
                )

                # Backpropagate actor loss
                actor_grads = tape.gradient(actor_loss, actor.trainable_variables)
                actor_optimizer.apply_gradients(zip(actor_grads, actor.trainable_variables))

            # Accumulate losses for logging
            total_actor_loss += tf.reduce_sum(actor_loss)
            total_critic_loss += critic_loss.numpy()
            update_steps += 1

    # Update running reward
    running_reward = 0.05 * episode_reward + (1 - 0.05) * running_reward

    # Decay learning rates
    actor_optimizer.learning_rate = max(initial_actor_lr * (lr_decay_rate ** episode), min_actor_lr)
    critic_optimizer.learning_rate = max(initial_critic_lr * (lr_decay_rate ** episode), min_critic_lr)


    # Log details
    avg_critic_loss = total_critic_loss / max(update_steps, 1)
    avg_actor_loss = total_actor_loss / max(update_steps, 1)

    template = (
        "Search epsilon: {:.2f}, "
        "Rand best obj: {:.2f}, "
        "Best obj {:.2f}, "
        "Running reward: {:.2f}, "
        "Critic loss: {:.4f}, "
        "Actor loss: {:.4f}, "
        "Episode final state: {:.2f}"
    )
    print(template.format(
        search_epsilon,
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
