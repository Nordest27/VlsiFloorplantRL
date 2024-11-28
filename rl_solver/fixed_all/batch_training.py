from collections import deque
import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Input, Dense, Flatten, LeakyReLU
from tensorflow.keras.applications import EfficientNetB0
from tensorflow.keras import Model
from tensorflow import keras, GradientTape
import random
import os

from tensorflow.python.framework.c_api_util import tf_buffer

from floorplant_gym_env import FloorPlantEnv
from bisect import bisect


print("Num GPUs Available: ",
    len(tf.config.experimental.list_physical_devices('GPU')))
# # Configure TensorFlow to use mixed precision for faster computation
# tf.keras.mixed_precision.set_global_policy('mixed_float16')

# Set threading options
tf.config.threading.set_inter_op_parallelism_threads(16)  # Adjust based on your CPU cores
tf.config.threading.set_intra_op_parallelism_threads(16)


# Initialize environment
env = FloorPlantEnv(16)

n_moves = 9
weighted_connections_size = env.n*(env.n-1)//2
# Define model parameters
hidden_nodes = 2**9

# Define EfficientNetB0 as the feature extractor
effnet_base = EfficientNetB0(include_top=False, input_shape=(max(env.n, 32), max(env.n, 32), 3))  # Use EfficientNet with no top layer
effnet_base.trainable = True

# Input layer for floorplan images
actor_input_layer = Input(shape=(max(env.n, 32), max(env.n, 32), 3))  # Input shape for the images

# Get the features from EfficientNetB0 (this is the backbone)
effnet_output = effnet_base(actor_input_layer)

# Flatten the output of EfficientNet to feed into the actor and critic
effnet_output_flattened = Flatten()(effnet_output)

# Action outputs
wfa = Dense(env.n, activation="softmax", name="wfa")(effnet_output_flattened)
wsa = Dense(env.n, activation="softmax", name="wsa")(effnet_output_flattened)
wma = Dense(n_moves, activation="softmax", name="wma")(effnet_output_flattened)

# Value output
critic = Dense(1, name="critic")(effnet_output_flattened)

# Combine the actor and critic outputs into a single model
model = Model(inputs=actor_input_layer, outputs=[wfa, wsa, wma, critic])

# Compile the model with an optimizer
optimizer = tf.keras.optimizers.Adam(learning_rate=1e-3)

# Display model summary (optional)
model.summary()

# Training parameters
batch_size = 128
sampling_batch = 8
env.max_steps = batch_size

gamma = 1.0
adv_lambda = 0.5
entropy_coefficient = 1e-1
eps = np.finfo(np.float32).eps.item()


def update_model(states, actions, rewards, next_states):

    with GradientTape() as tape:
        critic_values = tf.squeeze(model(states)[3], axis=-1)
        critic_next_values = tf.squeeze(model(next_states)[3], axis=-1)
        target_values = rewards + gamma * critic_next_values
        #target_values = rewards
        advantages = target_values - critic_values
        #advantages = (advantages - tf.reduce_mean(advantages)) / (tf.math.reduce_std(advantages) + 1e-8)
        critic_loss = 0.5*tf.reduce_mean(advantages ** 2)

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
        actor_loss_wfp += entropy_coefficient * tf.reduce_mean(predicted_wfp * tf.math.log(predicted_wfp + eps))
        actor_loss_wsp += entropy_coefficient * tf.reduce_mean(predicted_wsp * tf.math.log(predicted_wsp + eps))
        actor_loss_wmp += entropy_coefficient * tf.reduce_mean(predicted_wmp * tf.math.log(predicted_wmp + eps))

        # Combine all losses
        actor_loss = (
            actor_loss_wfp,
            actor_loss_wsp,
            actor_loss_wmp,
        )

    # Backpropagate loss
    grads = tape.gradient((*actor_loss, tf.constant(0)), model.trainable_variables)
    optimizer.apply_gradients(zip(grads, model.trainable_variables))


    return actor_loss, critic_loss

def main():
    # Training parameters
    buffer = deque(maxlen=batch_size)
    num_episodes = 10000

    # Training loop
    running_reward = 0
    for episode in range(num_episodes):

        sps_to_expand = []
        states_to_expand = []
        rand_sp_to_expand = []

        for i in range(sampling_batch):
            env.reset()
            state = env.draw(env.fpp)
            #state = np.ravel(np.append(env.flattened_observation()[0], env.flattened_observation()[1]))
            sps_to_expand.append((env.fpp.x(), env.fpp.y()))
            states_to_expand.append(state)
            rand_sp_to_expand.append((env.rand_fpp.x(), env.rand_fpp.y()))


        expanded_states = set()
        rand_expanded_states = set()

        if episode % 10 == 9:
            print("Simulated Annealing solution:", env.sa_fpp.get_current_sp_objective())
            env.sa_fpp.visualize()
            env.best_fpp.visualize()
            pass
            """
            inp = tf.expand_dims(tf.convert_to_tensor(state, dtype=tf.float32), 0)
            wfp, wsp, wmp, critic_value = model(inp)

            # Action sampling
            wfp_dist = wfp.numpy()[0]
            wsp_dist = wsp.numpy()[0]
            wmp_dist = wmp.numpy()[0]

            first_choice = np.argmax(wfp_dist)

            wsp_dist[first_choice] = 0
            wsp_dist /= sum(wsp_dist)
            second_choice = np.argmax(wsp_dist)

            move = np.argmax(wmp_dist)

            action = (first_choice, second_choice, move)
            print(f"Applied move {action}, "
                  f"with probabilities {wfp_dist[first_choice]}, {wsp_dist[second_choice]}, {wmp_dist[move]}")
            env.step(action)
            env.ini_fpp = env.fpp
            env.rand_ini_fpp = env.rand_fpp
            """

        episode_reward = 0
        done = False
        while not done:
            sp_indices = np.random.choice(len(states_to_expand), sampling_batch, replace=False)
            states = [states_to_expand[i] for i in sp_indices]

            inp = tf.convert_to_tensor(states, dtype=tf.float32)
            lwfp, lwsp, lwmp, _ = model(inp)

            for bi in range(len(sp_indices)):
                sp_i = sp_indices[bi]
                env.fpp.set_sp(*sps_to_expand[sp_i])
                env.rand_fpp.set_sp(*rand_sp_to_expand[random.randint(0, len(rand_sp_to_expand)-1)])
                expanded_states.add(str((env.fpp.x(), env.fpp.y())))
                rand_expanded_states.add(str((env.rand_fpp.x(), env.rand_fpp.y())))

                # Action sampling
                wfp_dist = lwfp[bi].numpy()
                wsp_dist = lwsp[bi].numpy()
                wmp_dist = lwmp[bi].numpy()
                """
                first_choice, second_choice, move = -1, -1, -1
                new_state_found = False
                prev_sp = (env.fpp.x(), env.fpp.y())
                while not new_state_found:
                    env.fpp.set_sp(*prev_sp)

                    if first_choice != -1 and random.random() < 0.33:
                        wfp_dist[first_choice] = eps
                    wfp_dist /= sum(wfp_dist)
                """

                first_choice = np.random.choice(env.n, p=wfp_dist)

                """
                    if second_choice != -1 and wfp_dist[first_choice] > eps and random.random() < 0.66:
                        wsp_dist[second_choice] = eps
                """
                wsp_dist[first_choice] = eps

                wsp_dist /= sum(wsp_dist)
                second_choice = np.random.choice(env.n, p=wsp_dist)

                """
                    if move != -1 and wfp_dist[first_choice] > eps and wsp_dist[second_choice] > eps:
                        wmp_dist[move] = eps
                    wmp_dist /= sum(wmp_dist)
                """
                move = np.random.choice(n_moves, p=wmp_dist)

                action = (first_choice, second_choice, move)
                _, reward, done, _ = env.step(action)
                """
                    if str((env.fpp.x(), env.fpp.y())) not in expanded_states:
                        new_state_found = True
                    else:
                        pass
                        #print(first_choice, second_choice, move)
                        #print("Repeated state")
                """

                if episode % 10 == 8 and bi == 0:
                    print(first_choice, second_choice, move)

                next_state = env.draw(env.fpp)
                #next_state = np.ravel(np.append(env.flattened_observation()[0], env.flattened_observation()[1]))

                # Store transition in replay buffer
                episode_reward += reward
                buffer.append((state, action, reward, next_state))

                if str((env.fpp.x(), env.fpp.y())) not in expanded_states:# and reward > 0:
                    sps_to_expand.append((env.fpp.x(), env.fpp.y()))
                    states_to_expand.append(next_state)
                if str((env.rand_fpp.x(), env.rand_fpp.y())) not in rand_expanded_states: # and (reward > 0 or random.random() > 0.75):
                    rand_sp_to_expand.append((env.rand_fpp.x(), env.rand_fpp.y()))

            """
            #for bi in range(sampling_batch):
            #    batch_rewards[bi] = calculate_returns(batch_rewards[bi], [False]*episode_length)
            
            buffer = []
            for bi in range(sampling_batch):
                for ei in range(episode_length):
                    buffer.append(
                        (batch_states[bi][ei], batch_actions[bi][ei], batch_rewards[bi][ei], batch_next_states[bi][ei])
                    )
            """
            # Perform batch updates
            if len(buffer) >= batch_size:
                states, actions, rewards, next_states = zip(*buffer)
                # Convert batch to tensors
                states = tf.convert_to_tensor(np.array(states), dtype=tf.float32)
                next_states = tf.convert_to_tensor(np.array(next_states), dtype=tf.float32)
                rewards = tf.convert_to_tensor(rewards, dtype=tf.float32)
                # Update
                actor_loss, critic_loss = update_model(states, actions, rewards, next_states)
                # Accumulate losses for logging
                total_actor_loss = tf.reduce_sum(actor_loss)
                total_critic_loss = critic_loss.numpy()
                buffer = deque(maxlen=batch_size)
                template = (
                    "Episode: {}, "
                    "Rand best obj: {:.2f}, "
                    "Best obj {:.2f}, "
                    "Running reward: {:.2f}, "
                    "Episode reward: {:.2f}, "
                    "Critic loss: {:.4f}, "
                    "Actor loss: {:.4f}, "
                    "Episode final state: {:.2f}, "
                    "last move: {}"
                )

                print(template.format(
                    episode,
                    env.rand_best_fpp.get_current_sp_objective(),
                    env.best_obj,
                    running_reward,
                    episode_reward,
                    total_critic_loss,
                    total_actor_loss,
                    env.fpp.get_current_sp_objective(),
                    (first_choice, second_choice, move)
                ))
            #buffer = deque(maxlen=batch_size)

        # Update running reward
        if running_reward != 0:
            running_reward = 0.05 * episode_reward + (1 - 0.05) * running_reward
        else:
            running_reward = episode_reward

        #entropy_coefficient = max(entropy_coefficient*0.99, 1e-6)

    env.close()

if __name__ == "__main__":
    main()