# using example https://keras.io/examples/rl/actor_critic_cartpole/
import numpy as np
import os
os.environ["KERAS_BACKEND"] = "tensorflow"
from tensorflow import keras, GradientTape, math, square
import gym
from floorplant_gym_env import FloorPlantEnv
import numpy as np


# Create the CartPole Environment
env = FloorPlantEnv(5)

# Define the actor and critic networks
input_layer = keras.layers.Input((env.n*env.n + env.n*env.n + env.n*4 + env.n*env.n,))
hidden_layer = keras.layers.LeakyReLU(negative_slope=0.25)(keras.layers.Dense(2048)(input_layer))
hidden_layer = keras.layers.LeakyReLU(negative_slope=0.25)(keras.layers.Dense(2048)(hidden_layer))
hidden_layer = keras.layers.LeakyReLU(negative_slope=0.25)(keras.layers.Dense(2048)(hidden_layer))

wfa = keras.layers.Dense(env.n, activation='softmax')(hidden_layer)
wsa = keras.layers.Dense(env.n, activation='softmax')(hidden_layer)
wma = keras.layers.Dense(10, activation='softmax')(hidden_layer)

critic = keras.layers.Dense(1)(hidden_layer)

model = keras.Model(inputs=input_layer, outputs=[wfa, wsa, wma, critic])
"""
num_hidden = 128

inputs = keras.layers.Input((env.n + env.n + env.n*4 + env.n*env.n,))
common = keras.layers.Dense(num_hidden, activation="tanh")(inputs)
wfa = keras.layers.Dense(env.n, activation="softmax")(common)
wsa = keras.layers.Dense(env.n, activation="softmax")(common)
wma = keras.layers.Dense(10, activation="softmax")(common)
critic = keras.layers.Dense(1)(common)

model = keras.Model(inputs=inputs, outputs=[wfa, wsa, wma, critic])
"""
# Define optimizer and loss functions
optimizer = keras.optimizers.Adam(learning_rate=1e-5)
huber_loss = keras.losses.Huber()

# Main training loop
num_episodes = 1000000
gamma = 0.99
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
        for t in range(1, 2+int(np.log(1+episode))):  # Limit the number of time steps

            # Predict action probabilities and estimated future rewards
            # from environment state
            wfp, wsp, wmp, critic_value = model(
                keras.ops.expand_dims(
                    keras.ops.convert_to_tensor(env.flattened_observation()), 0
                )
            )
            critic_value_history.append(critic_value[0, 0])

            # Sample action from action probability distributions
            first_choice = np.random.choice(env.n, p=wfp.numpy()[0])
            second_choice = np.random.choice(env.n, p=wsp.numpy()[0])
            move = np.random.choice(10, p=wmp.numpy()[0])
            action_probs_history.append((
                keras.ops.log(wfp[0, first_choice]),
                keras.ops.log(wsp[0, second_choice]),
                keras.ops.log(wmp[0, move])
            ))

            # Apply the sampled action in our environment
            _, reward, done, _= env.step((first_choice, second_choice, move))
            state = env.flattened_observation()
            rewards_history.append(reward)
            episode_reward += reward

            if episode_count % 100 == 0:
                env.fpp.visualize()
            if done:
                break

        # Update running reward
        running_reward = 0.05 * episode_reward + (1 - 0.05) * running_reward

        # Calculate expected value from rewards
        # - At each timestep what was the total reward received after that timestep
        # - Rewards in the past (future?) are discounted by multiplying them with gamma
        # - These are the labels for our critic
        returns = []
        discounted_sum = 0
        for r in rewards_history[::-1]:
            discounted_sum = r + gamma * discounted_sum
            returns.insert(0, discounted_sum)

        # Normalize
        returns = np.array(returns)
        returns = (returns - np.mean(returns)) / (np.std(returns) + eps)
        returns = returns.tolist()

        # Calculating loss values to update our network
        history = zip(action_probs_history, critic_value_history, returns)
        actor_losses = ([], [], [])
        critic_losses = []
        for (log_prob_wf, log_prob_ws, log_prob_wm), value, ret in history:
            # At this point in history, the critic estimated that we would get a
            # total reward = `value` in the future. We took an action with log probability
            # of `log_prob` and ended up receiving a total reward = `ret`.
            # The actor must be updated so that it predicts an action that leads to
            # high rewards (compared to critic's estimate) with high probability.
            diff = ret - value

            actor_losses[0].append(-log_prob_wf * diff)
            actor_losses[1].append(-log_prob_ws * diff)
            actor_losses[2].append(-log_prob_wm * diff)

            # The critic must be updated so that it predicts a better estimate of
            # the future rewards.
            critic_losses.append(
                huber_loss(keras.ops.expand_dims(value, 0), keras.ops.expand_dims(ret, 0))
            )

        # Backpropagation
        loss_value = [
            sum(actor_losses[0])/len(critic_losses),
            sum(actor_losses[1])/len(critic_losses),
            sum(actor_losses[2])/len(critic_losses),
            sum(critic_losses)/len(critic_losses)
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
        template = "running reward: {:.2f}, episode reward: {:.2f} and number of moves {}  at episode {}"
        print(template.format(running_reward, episode_reward, len(critic_losses), episode_count))


env.close()