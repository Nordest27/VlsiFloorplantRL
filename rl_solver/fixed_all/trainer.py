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
env = FloorPlantEnv(25)
n_moves = 9

# Define the actor and critic networks
input_layer = keras.layers.Input((env.n + env.n,))

hidden_layer = keras.layers.Embedding(env.n, 4)(input_layer)
hidden_layer = keras.layers.Flatten()(hidden_layer)
hidden_layer = keras.layers.LeakyReLU(negative_slope=0.1)(keras.layers.Dense(2048)(hidden_layer))
#hidden_layer = keras.layers.LeakyReLU(negative_slope=0.1)(keras.layers.Dense(512)(hidden_layer))


# wfa = keras.layers.Dense(env.n, activation='softmax')(hidden_layer)
# wsa = keras.layers.Dense(env.n, activation='softmax')(hidden_layer)
# wma = keras.layers.Dense(10, activation='softmax')(hidden_layer)
actor_hidden_layer = keras.layers.Dense(2048, activation="tanh")(hidden_layer)
actor = keras.layers.Dense(env.n*env.n*n_moves, activation='softmax')(actor_hidden_layer)

critic_hidden_layer = keras.layers.LeakyReLU(negative_slope=0.1)(keras.layers.Dense(2048)(hidden_layer))
critic = keras.layers.Dense(1)(critic_hidden_layer)

model = keras.Model(inputs=input_layer, outputs=[actor, critic])

# Define optimizer and loss functions
optimizer = keras.optimizers.Adam(learning_rate=1e-2, clipnorm=1.0)


# Main training loop
num_episodes = 1000000
gamma = 1
eps = np.finfo(np.float32).eps.item()
action_probs_history = []
critic_value_history = []
returns_history = []
running_reward = 0
episode_count = 0

batch_size = 20

env.reset()

for episode in range(num_episodes):
    episode_count += 1

    # print("/////////////////////////////////////////////////////")
    # print("New Env state")
    # for v in env.observation:
        #print(v)
    state = env.flattened_observation()
    # env.fpp.visualize()
    episode_reward = 0

    if episode_count % 50 == 0:
        env.reset()
        action_probs, _ = model(
            keras.ops.expand_dims(
                keras.ops.convert_to_tensor(env.flattened_observation()), 0
            )
        )
        action = np.argmax(action_probs.numpy()[0])
        first_choice, second_choice, move = (
            int(action/(env.n*n_moves)),
            int((action/n_moves)%env.n),
            action%n_moves
        )
        env.step((
            first_choice,
            second_choice,
            move
        ))

    with GradientTape() as tape:
        if episode_count % 10 == 0:
            pass
            env.best_fpp.visualize_sa_solution()
            env.best_fpp.visualize()


        for batch_i in range(batch_size):
            env.reset()
            rewards = []
            while True: #range(2+int(np.log2(1+episode))):  # Limit the number of time steps

                # Predict action probabilities and estimated future rewards
                # from environment state
                action_probs, critic_value = model(
                    keras.ops.expand_dims(
                        keras.ops.convert_to_tensor(env.flattened_observation()), 0
                    )
                )
                critic_value_history.append(critic_value[0, 0])

                # Sample action from action probability distributions
                epsilon = 0.05  # Exploration factor
                if np.random.rand() < epsilon:
                    action = np.random.choice(env.n * env.n * n_moves)  # Random action
                else:
                    action = np.random.choice(env.n * env.n * n_moves, p=action_probs.numpy()[0])  # Best action

                action_probs_history.append(keras.ops.log(action_probs[0, action] + eps))

                first_choice, second_choice, move = (
                    int(action/(env.n*n_moves)),
                    int((action/n_moves)%env.n),
                    action%n_moves
                )
                # Apply the sampled action in our environment
                _, reward, done, _= env.think_step((first_choice, second_choice, move))
                state = env.flattened_observation()
                rewards.append(reward)
                episode_reward += reward/batch_size

                #if episode_count % 100 == 0:
                    #print(rewards_history)
                    #print(f"action taken (fc: {first_choice}, sc: {second_choice}, m: {move})")
                    #env.fpp.visualize()

                if done:
                    break

            # Calculate expected value from rewards
            # - At each timestep what was the total reward received after that timestep
            # - Rewards in the past are discounted by multiplying them with gamma
            # - These are the labels for our critic
            returns = []
            discounted_sum = 0
            for r in rewards[::-1]:
                discounted_sum = r + gamma * discounted_sum
                returns.insert(0, discounted_sum)

            returns_history.extend(returns)

        # Update running reward
        running_reward = 0.05 * episode_reward + (1 - 0.05) * running_reward

        # Normalize
        returns_history = np.array(returns_history)
        if np.std(returns_history) > eps:
            returns_history = (returns_history - np.mean(returns_history)) / (np.std(returns_history) + eps)
        else:
            returns_history = returns_history - np.mean(returns_history)
        returns_history = returns_history.tolist()

        # Calculating loss values to update our network
        #print(returns_history)
        #print(critic_value_history)
        history = zip(action_probs_history, critic_value_history, returns_history)
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
            critic_losses.append(diff ** 2)

        # Backpropagation
        loss_value = [
            sum(actor_losses)/len(actor_losses),
            sum(critic_losses)/len(critic_losses)
        ]
        #print("Loss value: ", loss_value)
        grads = tape.gradient(loss_value, model.trainable_variables)
        optimizer.apply_gradients(zip(grads, model.trainable_variables))


        # Log details
        template = ("SA temp: {:.2f}, "
                    "best objective: {:.2f}, "
                    "actual objective: {:.2f}, "
                    "running reward: {:.2f}, "
                    "critic error: {:.2f}, "
                    "actors error: {:.2f}, "
                    "episode reward: {:.2f} "
                    "and number of moves {}  at episode {}")
        print(template.format(
            env.sa_temp,
            abs(env.best_obj),
            env.current_fpp.get_current_sp_objective(),
            running_reward,
            np.mean(critic_losses),
            np.mean(actor_losses),
            episode_reward,
            len(critic_losses)/batch_size,
            episode_count)
        )

        # Clear the loss and reward history
        action_probs_history.clear()
        critic_value_history.clear()
        returns_history.clear()

env.close()