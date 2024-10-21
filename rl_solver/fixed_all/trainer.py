# using example https://keras.io/examples/rl/actor_critic_cartpole/
import numpy as np
import os
#os.environ["KERAS_BACKEND"] = "tensorflow"
from keras import backend
from tensorflow import keras, GradientTape, math, square, compat
from tensorflow.python.keras.backend import set_session
# adjust values to your needs
config = compat.v1.ConfigProto( device_count = {'GPU': 1 , 'CPU': 4} )
sess = compat.v1.Session(config=config)
set_session(sess)

import gym
from floorplant_gym_env import FloorPlantEnv
import numpy as np


# Create the CartPole Environment
env = FloorPlantEnv(10)
n_moves = 9

# Define the actor and critic networks
input_layer = keras.layers.Input((env.n + env.n,))

hidden_layer = keras.layers.Embedding(env.n, 8)(input_layer)
hidden_layer = keras.layers.Flatten()(hidden_layer)
hidden_layer = keras.layers.Dense(2024, activation="tanh")(hidden_layer)
#hidden_layer = keras.layers.LeakyReLU(negative_slope=0.05)(keras.layers.Dense(2024)(hidden_layer))
#hidden_layer = keras.layers.LeakyReLU(negative_slope=0.05)(keras.layers.Dense(2024)(hidden_layer))
#hidden_layer = keras.layers.LeakyReLU(negative_slope=0.05)(keras.layers.Dense(2024)(hidden_layer))

# wfa = keras.layers.Dense(env.n, activation='softmax')(hidden_layer)
# wsa = keras.layers.Dense(env.n, activation='softmax')(hidden_layer)
# wma = keras.layers.Dense(10, activation='softmax')(hidden_layer)

actor = keras.layers.Dense(env.n*env.n*n_moves, activation='softmax')(hidden_layer)
critic = keras.layers.Dense(1)(hidden_layer)

model = keras.Model(inputs=input_layer, outputs=[actor, critic])

# Define optimizer and loss functions
optimizer = keras.optimizers.Adam(learning_rate=3e-4)

# Main training loop
num_episodes = 1000000
gamma = 1
eps = np.finfo(np.float32).eps.item()
action_probs_history = []
critic_value_history = []
rewards_history = []
running_reward = 0
episode_count = 0


env.reset()

for episode in range(num_episodes):
    env.reset()
    # print("/////////////////////////////////////////////////////")
    # print("New Env state")
    # for v in env.observation:
        #print(v)
    state = env.flattened_observation()
    # env.fpp.visualize()
    episode_reward = 0

    with GradientTape() as tape:
        if episode_count % 100 == 0:
            env.fpp.visualize()
        for t in range(1, 100):# 2+int(np.log(1+episode))):  # Limit the number of time steps

            # Predict action probabilities and estimated future rewards
            # from environment state
            action_probs, critic_value = model(
                keras.ops.expand_dims(
                    keras.ops.convert_to_tensor(env.flattened_observation()), 0
                )
            )
            critic_value_history.append(critic_value[0, 0])

            # Sample action from action probability distributions
            action = np.random.choice(env.n*env.n*n_moves
                                      , p=action_probs.numpy()[0])
            action_probs_history.append(
                keras.ops.log(action_probs[0, action]),
            )
            first_choice, second_choice, move = int(action/(env.n*n_moves)), int((action/n_moves)%env.n), action%n_moves
            # Apply the sampled action in our environment
            _, reward, done, _= env.step((first_choice, second_choice, move))
            state = env.flattened_observation()
            rewards_history.append(reward)
            episode_reward += reward

            if episode_count % 100 == 0:
                print(rewards_history)
                print(f"action taken (fc: {first_choice}, sc: {second_choice}, m: {move})")
                env.fpp.visualize()

            if done:
                break

        # Update running reward
        running_reward = 0.05 * episode_reward + (1 - 0.05) * running_reward

        # Calculate expected value from rewards
        # - At each timestep what was the total reward received after that timestep
        # - Rewards in the past are discounted by multiplying them with gamma
        # - These are the labels for our critic
        returns = []
        discounted_sum = 0
        for r in rewards_history[::-1]:
            discounted_sum = r + gamma * discounted_sum
            returns.insert(0, discounted_sum)

        # Normalize
        returns = np.array(returns)
        returns = (returns - np.mean(returns))
        returns = returns.tolist()

        # Calculating loss values to update our network
        history = zip(action_probs_history, critic_value_history, returns)
        actor_losses = []
        critic_losses = []
        for log_prob, value, ret in history:
            # At this point in history, the critic estimated that we would get a
            # total reward = `value` in the future. We took an action with log probability
            # of `log_prob` and ended up receiving a total reward = `ret`.
            # The actor must be updated so that it predicts an action that leads to
            # high rewards (compared to critic's estimate) with high probability.
            diff = ret - value

            actor_losses.append(-log_prob * diff)

            # The critic must be updated so that it predicts a better estimate of
            # the future rewards.
            critic_losses.append(
                keras.losses.MeanSquaredError()(
                    keras.ops.expand_dims(value, 0),
                    keras.ops.expand_dims(ret, 0)
                )
            )

        # Backpropagation
        loss_value = [
            sum(actor_losses),
            sum(critic_losses)
        ]
        #print("Loss value: ", loss_value)
        grads = tape.gradient(loss_value, model.trainable_variables)
        optimizer.apply_gradients(zip(grads, model.trainable_variables))

        # Clear the loss and reward history
        action_probs_history.clear()
        critic_value_history.clear()
        rewards_history.clear()

    # Log details
    episode_count += 1
    if episode_count % 1 == 0:
        template = ("SA temp: {:.2f}, "
                    "actual objective: {:.2f}, "
                    "running reward: {:.2f}, "
                    "critic error: {:.2f}, "
                    "actors error: {:.2f}, "
                    "episode reward: {:.2f} "
                    "and number of moves {}  at episode {}")
        print(template.format(
            env.sa_temp,
            abs(env.best_obj),
            running_reward,
            np.mean(critic_losses),
            np.mean(actor_losses),
            episode_reward,
            len(critic_losses),
            episode_count))


env.close()