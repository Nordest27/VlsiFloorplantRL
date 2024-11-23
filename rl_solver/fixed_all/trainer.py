# using example https://keras.io/examples/rl/actor_critic_cartpole/
import numpy as np
import os
#os.environ["KERAS_BACKEND"] = "tensorflow"
from keras import backend
import random
from tensorflow import keras, GradientTape, math, square, compat, constant, reduce_mean, reduce_sum
from tensorflow.python.keras.backend import set_session
# adjust values to your needs
config = compat.v1.ConfigProto( device_count = {'GPU': 1 , 'CPU': 4} )
sess = compat.v1.Session(config=config)
set_session(sess)


import gym
from floorplant_gym_env import FloorPlantEnv
import numpy as np

from numpy.random import normal
def add_noise(model):
    weights = model.get_weights()
    for layer in weights:
        noise = normal(loc=0.0, scale=1, size=layer.shape)
        layer += noise
    model.set_weights(weights)


# Create the CartPole Environment
env = FloorPlantEnv(10)
weighted_connections_size = env.n*(env.n-1)//2

n_moves = 9

hidden_nodes = 2**5

# Define the actor and critic networks
def create_model(output, actor: bool, extra_inps: int = 0):
    # Input for embeddings
    embeds_input_layer = keras.layers.Input((env.n * 2 + extra_inps,))
    embeds_hidden_layer = embeds_input_layer
#     embeds_hidden_layer = keras.layers.Embedding(env.n, 4)(embeds_input_layer)
#     embeds_hidden_layer = keras.layers.Flatten()(embeds_hidden_layer)

    # Add intermediate layer for embeddings
    embeds_hidden_layer = keras.layers.Dense(
        hidden_nodes, activation='relu', name='embed_dense_layer'
    )(embeds_hidden_layer)

    # Input for real-valued data
    reals_input_layer = keras.layers.Input((env.n * 2 + weighted_connections_size,))
    reals_hidden_layer = reals_input_layer
    # Add intermediate layer for real-valued inputs
    reals_hidden_layer = keras.layers.Dense(
        hidden_nodes, activation='relu', name='reals_dense_layer'
    )(reals_input_layer)

    # Combine embeddings and real-valued inputs
    hidden_layer = keras.layers.Concatenate()([embeds_hidden_layer, reals_hidden_layer])
#
#     # Add additional dense layers after concatenation (optional)
    hidden_layer = keras.layers.Dense(2*hidden_nodes, activation='relu', name='combined_dense_layer')(hidden_layer)
#
#     # Optional layers for actor-specific processing
#     if actor:
#         hidden_layer = keras.layers.Dense(2*hidden_nodes, activation='tanh', name='actor_dense_layer')(hidden_layer)

    return keras.Model(
        inputs={"xy": embeds_input_layer, "weights": reals_input_layer},
        outputs=output(hidden_layer)
    )


actor_wfa = create_model(keras.layers.Dense(env.n, activation='softmax'), True)
# mask_wfa = create_model(keras.layers.Dense(env.n, activation='sigmoid'), True)
actor_wsa = create_model(keras.layers.Dense(env.n, activation='softmax'), True, extra_inps=1)
# mask_wsa = create_model(keras.layers.Dense(env.n, activation='sigmoid'), True, extra_inps=1)
actor_wma = create_model(keras.layers.Dense(n_moves, activation='softmax'), True, extra_inps=2)
# mask_wma = create_model(keras.layers.Dense(env.n, activation='sigmoid'), True, extra_inps=1)
# actor_simple = create_model(keras.layers.Dense(2, activation='softmax'), True)
critic =  create_model(keras.layers.Dense(1), False)

# Define optimizer and loss functions
actor_wfa_optimizer = keras.optimizers.Adam(learning_rate=1e-4)
actor_wsa_optimizer = keras.optimizers.Adam(learning_rate=1e-4)
actor_wma_optimizer = keras.optimizers.Adam(learning_rate=1e-4)
critic_optimizer = keras.optimizers.Adam(learning_rate=1e-3)


search_epsilon = 0.5

# Main training loop
num_episodes = 1000000
eps = np.finfo(np.float32).eps.item()
action_probs_history = []
critic_value_history = []
rewards_history = []
running_reward = 0
episode_count = 0
gamma = 1

# Define regularization strength (hyperparameter)
entropy_strength = 1e-3# Adjust as needed


actor_losses = []

env.reset()

for episode in range(num_episodes):
    # print("/////////////////////////////////////////////////////")
    # print("New Env state")
    # for v in env.observation:
        #print(v)
    # env.fpp.visualize()
    max_episode_reward = -np.inf
    episode_reward = 0
    episode_reward_index = 0
    episode_rewards = []

    actor_losses = ([], [], [])
    critic_losses = []

#     if episode%100 == 0: #np.mean(np.abs(actor_losses)) < 0.05 and search_epsilon < 0.05:
# #        print("Actor stagnated, assigning best_sp")
# #        search_epsilon = max((0.999**episode)*0.5, 0.1)
    if episode % 100 == 99:
        print("Simulated Annealing solution:", env.sa_fpp.get_current_sp_objective())
        env.sa_fpp.visualize()
        env.best_fpp.visualize()
        pass
#         env.reset()
#         state = env.flattened_observation()
#         inp = keras.ops.expand_dims(keras.ops.convert_to_tensor(state), 0)
#         critic_value = critic(inp)
#
#         wfp = actor_wfa(inp)
#         first_choice = np.random.choice(env.n, p=wfp.numpy()[0])
#         state = np.append(state, first_choice)
#         inp = keras.ops.expand_dims(keras.ops.convert_to_tensor(state), 0)
#
#         wsp = actor_wsa(inp)
#         wsp_dist = wsp.numpy()[0]
#         wsp_dist[first_choice] = 0
#         wsp_dist += eps
#         wsp_dist /= sum(wsp_dist)
#         second_choice = np.random.choice(env.n, p=wsp_dist)
#         state = np.append(state, second_choice)
#         inp = keras.ops.expand_dims(keras.ops.convert_to_tensor(state), 0)
#
#         wmp = actor_wma(inp)
#         move = np.random.choice(n_moves, p=wmp.numpy()[0])
#
#         env.step((first_choice, second_choice, move), just_step=True)
#         env.ini_fpp = env.fpp
#         env.rand_ini_fpp = env.rand_fpp
#         env.ini_fpp = env.best_fpp.copy()
#         env.rand_ini_fpp = env.rand_best_fpp.copy()
#         print(f"Action taken (fc: {first_choice}, sc: {second_choice}, m: {move}), "
#               f"wfp: {wfp[0, first_choice]}, wsp: {wsp[0, second_choice]}, wmp: {wmp[0, move]}")
#         add_noise(actor_wfa)
#         add_noise(actor_wsa)
#         add_noise(actor_wma)
#         env.ini_fpp = env.best_fpp.copy()
#         env.rand_ini_fpp = env.rand_best_fpp.copy()

    env.reset()
    i = 0
    while True: #range(2+int(np.log2(1+episode))):  # Limit the number of time steps
        # Predict action probabilities and estimated future rewards
        # from environment state
        critic_tape = GradientTape()
        wfa_tape = GradientTape()
        wsa_tape = GradientTape()
        wma_tape = GradientTape()

        state_xy, state_weights = env.flattened_observation()
        xy_inp = keras.ops.expand_dims(keras.ops.convert_to_tensor(state_xy), 0)
        weights_inp = keras.ops.expand_dims(keras.ops.convert_to_tensor(state_weights), 0)

        with critic_tape:
            critic_value = critic({"xy": xy_inp, "weights": weights_inp})
        with wfa_tape:
            wfp = actor_wfa({"xy": xy_inp, "weights": weights_inp})
        first_choice = np.random.choice(env.n, p=wfp.numpy()[0])
        state_xy = np.append(state_xy, first_choice)
        xy_inp = keras.ops.expand_dims(keras.ops.convert_to_tensor(state_xy), 0)
        with wsa_tape:
            wsp = actor_wsa({"xy": xy_inp, "weights": weights_inp})
        wsp_dist = wsp.numpy()[0]
        wsp_dist[first_choice] = 0
        wsp_dist += eps
        wsp_dist /= sum(wsp_dist)
        second_choice = np.random.choice(env.n, p=wsp_dist)
        state_xy = np.append(state_xy, second_choice)
        xy_inp = keras.ops.expand_dims(keras.ops.convert_to_tensor(state_xy), 0)
        with wma_tape:
            wmp = actor_wma({"xy": xy_inp, "weights": weights_inp})
        move = np.random.choice(n_moves, p=wmp.numpy()[0])

        print(first_choice, second_choice, move)

        critic_value_history.append(critic_value[0, 0])

        if episode % 100 == 98:
            print(f"Action taken (fc: {first_choice}, sc: {second_choice}, m: {move}), "
                  f"wfp: {wfp[0, first_choice]}, wsp: {wsp[0, second_choice]}, wmp: {wmp[0, move]}")
#                 print("Previous")
#                 env.fpp.visualize()

        # Apply the sampled action in our environment
        _, reward, done, _= env.step((first_choice, second_choice, move))
        print(reward)

        state_xy, state_weights = env.flattened_observation()
        xy_inp = keras.ops.expand_dims(keras.ops.convert_to_tensor(state_xy), 0)
        weights_inp = keras.ops.expand_dims(keras.ops.convert_to_tensor(state_weights), 0)
        critic_next_state_value = critic({"xy": xy_inp, "weights": weights_inp})

        episode_rewards.append(reward)
        episode_reward += reward
        if episode_reward < reward:
            max_episode_reward = reward
            episode_reward_index = i

        # Add L2 regularization
        """
        regularization_loss = regularization_strength * sum(
            reduce_sum(square(var)) for var in critic.trainable_variables
        )
        critic_loss += regularization_loss  # Combine the original loss with the regularization term
        """

        # Regularize to encourage exploration
        with critic_tape:
            advantage = reward + (1.0 - done) * gamma * critic_next_state_value - critic_value
            critic_loss = advantage**2#*(1.0 if advantage > 0 else 0)  # Original loss term
        with wfa_tape:
            wfa_log_prob = keras.ops.log(wfp[0, first_choice] + eps)
            wfa_entropy_loss = reduce_sum(wfp*keras.ops.log(wfp))
            wfa_loss = -wfa_log_prob * advantage + entropy_strength * wfa_entropy_loss
        with wsa_tape:
            wsa_log_prob = keras.ops.log(wsp[0, second_choice] + eps)
            wsa_entropy_loss = reduce_sum(wsp*keras.ops.log(wsp))
            wsa_loss = -wsa_log_prob * advantage + entropy_strength * wsa_entropy_loss
        with wma_tape:
            wma_log_prob = keras.ops.log(wmp[0, move] + eps)
            wma_entropy_loss = reduce_sum(wmp*keras.ops.log(wmp))
            wma_loss = -wma_log_prob * advantage + wma_entropy_loss


        action_probs_history.append((
            wfa_log_prob,
            wsa_log_prob,
            wma_log_prob
        ))

        critic_losses.append(critic_loss)
        actor_losses[0].append(wfa_loss)
        actor_losses[1].append(wsa_loss)
        actor_losses[2].append(wma_loss)

#             if random.random() < 0.98:
#                 env.steps -= 1
#                 env.fpp = prev_fpp

        critic_grads = critic_tape.gradient(critic_loss, critic.trainable_variables)
        critic_optimizer.apply_gradients(zip(critic_grads, critic.trainable_variables))

        actor_wfa_grads = wfa_tape.gradient(wfa_loss, actor_wfa.trainable_variables)
        actor_wfa_optimizer.apply_gradients(zip(actor_wfa_grads, actor_wfa.trainable_variables))

        actor_wsa_grads = wsa_tape.gradient(wsa_loss, actor_wsa.trainable_variables)
        actor_wsa_optimizer.apply_gradients(zip(actor_wsa_grads, actor_wsa.trainable_variables))

        actor_wma_grads = wma_tape.gradient(wma_loss, actor_wma.trainable_variables)
        actor_wma_optimizer.apply_gradients(zip(actor_wma_grads, actor_wma.trainable_variables))

        if episode % 100 == 98:
            print(f"After, state V: {critic_value}, next state V: {critic_next_state_value}, advantage {advantage}, "
                  f"Actor losses {(actor_losses[0][-1], actor_losses[1][-1], actor_losses[2][-1])}"
            )
#                 env.fpp.visualize()
        i += 1
        if done:
            break

    # Calculate expected value from rewards
    # - At each timestep what was the total reward received after that timestep
    # - Rewards in the past are discounted by multiplying them with gamma
    # - These are the labels for our critic
#         returns = []
#         discounted_sum = 0
#         for r in rewards[::-1]:
#             discounted_sum = r + gamma * discounted_sum
#             returns.insert(0, discounted_sum)
#         aux_rewards = [0 for i in range(len(episode_rewards))]
#         for i in range(len(episode_rewards)):
#             aux_rewards[i] = np.sum([0, *episode_rewards[i+1:]])
#         episode_rewards = aux_rewards

    # Update running reward
    running_reward = 0.05 * episode_reward + (1 - 0.05) * running_reward
    rewards_history.append(episode_reward)


    # Calculating loss values to update our network
    #print(returns_history)
    #print(critic_value_history)
#         action_probs_history = action_probs_history
#         critic_value_history = critic_value_history
#         history = zip(action_probs_history, [0] + critic_value_history, critic_value_history, [0] + episode_rewards)
#         i = 0
#         mean_val = reduce_mean(rewards_history[-10:])
#         # episode_reward_index = np.inf
#         for (log_prob_wf, log_prob_ws, log_prob_wm), prev_value, value, ret in history:
# #             if i < episode_reward_index:
# #                 mean_diff = (episode_reward - mean_val)
# #             else:
# #                 mean_diff = (reduce_mean(episode_rewards[i:]).numpy() - mean_val)
# #             mean_diff *= 0
#             diff = ret + value - prev_value
#
#             critic_losses.append(diff**2)
#             actor_losses[0].append(-log_prob_wf * diff)
#             actor_losses[1].append(-log_prob_ws * diff)
#             actor_losses[2].append(-log_prob_wm * diff)
# #             actor_losses[0].append(-log_prob_wf * (mean_diff * int(i <= episode_reward_index) + diff ))
# #             actor_losses[1].append(-log_prob_ws * (mean_diff * int(i <= episode_reward_index) + diff ))
# #             actor_losses[2].append(-log_prob_wm * (mean_diff * int(i <= episode_reward_index) + diff ))
#
#             i += 1
#

    # Backpropagation
#         loss_values = list(zip(*actor_losses, critic_losses))

    #print("Loss value: ", loss_value)
#         critic_grads = tape.gradient(critic_losses, critic.trainable_variables)
#         critic_optimizer.apply_gradients(zip(critic_grads, critic.trainable_variables))
#         actor_grads = tape.gradient(actor_losses, actor.trainable_variables)
#         actor_optimizer.apply_gradients(zip(actor_grads, actor.trainable_variables))


    # Log details
    template = (
        "Episode {}, "
        "Rand Ini obj {:.2f}, "
        "Ini obj {:.2f}, "
    )
    print(template.format(
        episode,
        env.rand_ini_fpp.get_current_sp_objective(),
        env.ini_fpp.get_current_sp_objective(),
    ))
    template = (
        "Search epsilon: {:.2f}, "
        "Rand best obj: {:.2f}, "
        "Best obj {:.2f}, "
        "running reward: {:.2f}, "
        "critic values: {:.2f}, "
        "critic error: {:.2f}, "
        "actors error: {:.2f}, "
        "actors absolute error: {:.2f}, "
        "episode final state: {:.2f}, "
        "episode reward: {:.2f}, "
        "rewards mean: {:.2f}, "
       )
    print(template.format(
        search_epsilon,
        env.rand_best_fpp.get_current_sp_objective(),
        env.best_obj,
        running_reward,
        np.mean(critic_value_history),
        np.mean(critic_losses),
        np.mean(actor_losses),
        np.mean(np.abs(actor_losses)),
        env.fpp.get_current_sp_objective(),
        np.mean(episode_reward),
        np.mean(rewards_history[-10:]),
    ))

    # Clear the loss and reward history

    action_probs_history.clear()
    critic_value_history.clear()

env.close()