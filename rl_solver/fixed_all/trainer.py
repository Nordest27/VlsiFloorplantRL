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


# Create the CartPole Environment
env = FloorPlantEnv(20)
n_moves = 3

hidden_nodes = 2**9

# Define the actor and critic networks
actor_input_layer = keras.layers.Input((env.n + env.n,))

actor_hidden_layer = keras.layers.Embedding(env.n, int(np.ceil(np.log2(env.n))))(actor_input_layer)
actor_hidden_layer = keras.layers.Flatten()(actor_hidden_layer)
actor_hidden_layer = keras.layers.LeakyReLU(negative_slope=0.1)(keras.layers.Dense(hidden_nodes)(actor_hidden_layer))
actor_hidden_layer = keras.layers.LeakyReLU(negative_slope=0.1)(keras.layers.Dense(hidden_nodes)(actor_hidden_layer))
actor_hidden_layer = keras.layers.Dense(hidden_nodes, activation="tanh")(actor_hidden_layer)

wfa = keras.layers.Dense(env.n, activation='softmax')(actor_hidden_layer)
wsa = keras.layers.Dense(env.n, activation='softmax')(actor_hidden_layer)
wma = keras.layers.Dense(n_moves, activation='softmax')(actor_hidden_layer)

actor = keras.Model(inputs=actor_input_layer, outputs=[wfa, wsa, wma])

# Define the actor and critic networks
critic_input_layer = keras.layers.Input((env.n + env.n,))

critic_hidden_layer = keras.layers.Embedding(env.n, int(np.ceil(np.log2(env.n))))(critic_input_layer)
critic_hidden_layer = keras.layers.Flatten()(critic_hidden_layer)
critic_hidden_layer = keras.layers.LeakyReLU(negative_slope=0.1)(keras.layers.Dense(hidden_nodes)(critic_hidden_layer))
critic_hidden_layer = keras.layers.LeakyReLU(negative_slope=0.1)(keras.layers.Dense(hidden_nodes)(critic_hidden_layer))
critic_output = keras.layers.Dense(1)(critic_hidden_layer)

critic = keras.Model(inputs=critic_input_layer, outputs=critic_output)

# Define optimizer and loss functions
actor_optimizer = keras.optimizers.Adam(learning_rate=1e-4)
critic_optimizer = keras.optimizers.Adam(learning_rate=1e-3)


search_epsilon = 1.0

# Main training loop
num_episodes = 10000
eps = np.finfo(np.float32).eps.item()
action_probs_history = []
critic_value_history = []
rewards_history = []
running_reward = 0
episode_count = 0
gamma = 1


actor_losses = []

env.reset()

for episode in range(num_episodes):
    episode_count += 1

    # print("/////////////////////////////////////////////////////")
    # print("New Env state")
    # for v in env.observation:
        #print(v)
    state = env.flattened_observation()
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
#        env.ini_fpp = env.best_fpp.copy()
#        env.rand_ini_fpp = env.rand_best_fpp.copy()

    with GradientTape(persistent=True) as tape:
        if episode_count % 100 == 0:
#             search_epsilon = 0.5
            print("Simulated Annealing solution:", env.sa_fpp.get_current_sp_objective())
            env.sa_fpp.visualize()
            env.best_fpp.visualize()
            env.reset()
            pass
#             first_choice = np.random.choice(env.n, p=wfp.numpy()[0])
#             wsp_dist = wsp.numpy()[0]
#             wsp_dist[first_choice] = 0
#             wsp_dist += eps
#             wsp_dist /= sum(wsp_dist)
#             second_choice = np.random.choice(env.n, p=wsp_dist)
#             move = np.random.choice(n_moves, p=wmp.numpy()[0])
#
#             env.step((first_choice, second_choice, move))
#             env.ini_fpp = env.fpp
#             env.rand_ini_fpp = env.rand_fpp

        env.reset()
        i = 0
        while True: #range(2+int(np.log2(1+episode))):  # Limit the number of time steps
            # Predict action probabilities and estimated future rewards
            # from environment state
            inp = keras.ops.expand_dims(
                keras.ops.convert_to_tensor(env.flattened_observation()), 0
            )
            critic_value = critic(inp)
            wfp, wsp, wmp = actor(inp)
            critic_value_history.append(critic_value[0, 0])

            # Sample action from action probability distributions
            if False and np.random.rand() < search_epsilon:
                first_choice, second_choice = np.random.choice(env.n, 2, replace=False)
                assert first_choice != second_choice
                move = np.random.choice(n_moves)
            else:
                wfp_dist = wfp.numpy()[0]
                wsp_dist = wsp.numpy()[0]
                wmp_dist = wmp.numpy()[0]

                wfp_dist += search_epsilon
                wsp_dist += search_epsilon
                wmp_dist += search_epsilon

                wfp_dist /= sum(wfp_dist)
                first_choice = np.random.choice(env.n, p=wfp_dist)

                wsp_dist[first_choice] = 0
                wsp_dist /= sum(wsp_dist)
                second_choice = np.random.choice(env.n, p=wsp_dist)

                wmp_dist /= sum(wmp_dist)
                move = np.random.choice(n_moves, p=wmp_dist)


            action_probs_history.append((
                keras.ops.log(wfp[0, first_choice] + eps),
                keras.ops.log(wsp[0, second_choice] + eps),
                keras.ops.log(wmp[0, move] + eps)
            ))

            if episode_count % 100 == 99:
                print(f"Action taken (fc: {first_choice}, sc: {second_choice}, m: {move}), "
                      f"wfp: {wfp[0, first_choice]}, wsp: {wsp[0, second_choice]}, wmp: {wmp[0, move]}")
                print("Previous")
                env.fpp.visualize()

            # Apply the sampled action in our environment
            _, reward, done, _= env.step((first_choice, second_choice, move))

            inp = keras.ops.expand_dims(
                keras.ops.convert_to_tensor(env.flattened_observation()), 0
            )
            critic_next_state_value = critic(inp)

            episode_rewards.append(reward)
            episode_reward += reward
            if episode_reward < reward:
                max_episode_reward = reward
                episode_reward_index = i

            # Define regularization strength (hyperparameter)
            regularization_strength = 1e-4  # Adjust as needed

            # Compute critic loss
            advantage = reward + (1.0 - done) * gamma * critic_next_state_value - critic_value
            critic_loss = advantage**2  # Original loss term
            critic_losses.append(critic_loss)
#
#             # Add L2 regularization
#             regularization_loss = regularization_strength * sum(
#                 reduce_sum(square(var)) for var in critic.trainable_variables
#             )
#             critic_loss += regularization_loss  # Combine the original loss with the regularization term
            # Regularize to encourage exploration
            entropy_loss = -regularization_strength * (
                reduce_sum(wfp * action_probs_history[-1][0]) +
                reduce_sum(wsp * action_probs_history[-1][1]) +
                reduce_sum(wmp * action_probs_history[-1][2])
            )

            actor_losses[0].append(-action_probs_history[-1][0] * advantage + entropy_loss)
            actor_losses[1].append(-action_probs_history[-1][1] * advantage + entropy_loss)
            actor_losses[2].append(-action_probs_history[-1][2] * advantage + entropy_loss)

#             if random.random() < 0.98:
#                 env.steps -= 1
#                 env.fpp = prev_fpp

            critic_grads = tape.gradient(critic_losses[-1], critic.trainable_variables)
            critic_optimizer.apply_gradients(zip(critic_grads, critic.trainable_variables))
            actor_grads = tape.gradient(
                (actor_losses[0][-1], actor_losses[1][-1], actor_losses[2][-1]),
                actor.trainable_variables)
            actor_optimizer.apply_gradients(zip(actor_grads, actor.trainable_variables))
            if episode_count % 100 == 99:
                print(f"After, state Q: {critic_value}, next state Q: {critic_next_state_value}, advantage {advantage}, "
                      f"Actor losses {(actor_losses[0][-1], actor_losses[1][-1], actor_losses[2][-1])}"
                )
                env.fpp.visualize()
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

        # Normalize
        """
        returns_history = np.array(returns_history)
        if np.std(returns_history) > eps:
            returns_history = (returns_history - np.mean(returns_history)) / (np.std(returns_history) + eps)
        else:
            returns_history = returns_history - np.mean(returns_history)
        returns_history = returns_history.tolist()
        """

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
            "Search epsilon: {:.2f}, "
            "Rand best obj: {:.2f}, "
            "Best obj {:.2f}, "
            "running reward: {:.2f}, "
            "critic values: {:.2f}, "
            "critic error: {:.2f}, "
            "actors error: {:.2f}, "
            "actors absolute error: {:.2f}, "
            "episode final state: {:.2f}, "
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
            np.mean(rewards_history[-10:]),
        ))

        # Clear the loss and reward history

        action_probs_history.clear()
        critic_value_history.clear()

        search_epsilon = max(search_epsilon*0.95, 0.001)

env.close()