# using example https://keras.io/examples/rl/actor_critic_cartpole/
import numpy as np
import os
#os.environ["KERAS_BACKEND"] = "tensorflow"
from keras import backend
from tensorflow import keras, GradientTape, math, square, compat
from tensorflow.python.keras.backend import set_session
from tensorflow.python.ops.gen_data_flow_ops import resource_accumulator_take_gradient

# adjust values to your needs
config = compat.v1.ConfigProto( device_count = {'GPU': 1 , 'CPU': 4} )
sess = compat.v1.Session(config=config)
set_session(sess)


import gym
from floorplant_gym_env import FloorPlantEnv
import numpy as np


# Create the CartPole Environment
env = FloorPlantEnv(20)
n_moves = 9

# Define the actor and critic networks
#offsets_input_layer = keras.layers.Input((env.n + env.n,))
x_connected_input_layer = keras.layers.Input((env.n, env.n, 1))
y_connected_input_layer = keras.layers.Input((env.n, env.n, 1))

# Convolutional layer for x_connected_input_layer
x_conv_layer = keras.layers.Conv2D(
    filters=8, kernel_size=3, padding='same', activation='relu'
)(x_connected_input_layer)
x_conv_layer = keras.layers.MaxPooling2D(pool_size=2)(x_conv_layer)
x_conv_layer = keras.layers.Flatten()(x_conv_layer)

# Convolutional layer for y_connected_input_layer
y_conv_layer = keras.layers.Conv2D(
    filters=8, kernel_size=3, padding='same', activation='relu'
)(y_connected_input_layer)
y_conv_layer = keras.layers.MaxPooling2D(pool_size=2)(y_conv_layer)
y_conv_layer = keras.layers.Flatten()(y_conv_layer)

hidden_layer = keras.layers.Concatenate()([x_conv_layer, y_conv_layer])
#hidden_layer = keras.layers.LeakyReLU(negative_slope=0.1)(keras.layers.Dense(512)(hidden_layer))


# wfa = keras.layers.Dense(env.n, activation='softmax')(hidden_layer)
# wsa = keras.layers.Dense(env.n, activation='softmax')(hidden_layer)
# wma = keras.layers.Dense(10, activation='softmax')(hidden_layer)
actor_hidden_layer = keras.layers.Dense(512, activation="relu")(hidden_layer)
actor = keras.layers.Dense(env.n*env.n*n_moves)(actor_hidden_layer)
actor = keras.layers.Softmax()(
    actor,
    mask=[not action//(env.n*n_moves) == (action//n_moves)%env.n for action in range(env.n * env.n * n_moves)]
)

critic_hidden_layer = keras.layers.LeakyReLU(negative_slope=0.1)(keras.layers.Dense(512)(hidden_layer))
critic = keras.layers.Dense(1)(critic_hidden_layer)

model = keras.Model(
    inputs=[x_connected_input_layer, y_connected_input_layer],
    outputs=[actor, critic]
)

# Define optimizer and loss functions
optimizer = keras.optimizers.Adam(learning_rate=1e-3)


# Main training loop
num_episodes = 1000000
gamma = 1
eps = np.finfo(np.float32).eps.item()
action_probs_history = []
critic_value_history = []
returns_history = []
running_reward = 0
episode_count = 0
max_reward_seen = 0

epsilon = 1.0  # Exploration factor
epsilon_decay = 0.95

batch_size = 10

env.reset()

for episode in range(num_episodes):
    episode_count += 1

    # print("/////////////////////////////////////////////////////")
    # print("New Env state")
    # for v in env.observation:
        #print(v)
    # env.fpp.visualize()
    episode_reward = 0

    with GradientTape() as tape:
        if episode_count % 10 == 0:
            pass
            env.best_fpp.visualize_sa_solution()
            env.best_fpp.visualize()

        for batch_i in range(batch_size):
            env.reset()
            max_episode_reward = 0
            rewards = []
            while True: #range(2+int(np.log2(1+episode))):  # Limit the number of time steps

                # Assuming get_input returns offsets, x_con, and y_con as described
                x_con, y_con = env.get_input()

                # Convert to NumPy arrays
                #offsets = np.array(offsets)  # Should be shape (n,) or (1, n) for a single batch
                x_con = np.array(x_con)  # Should be shape (height, width, channels)
                y_con = np.array(y_con)  # Should be shape (height, width, channels)

                # Expand dimensions to add a batch dimension if necessary
                x_con = np.expand_dims(x_con, axis=0)  # Shape (1, height, width, channels)
                y_con = np.expand_dims(y_con, axis=0)  # Shape (1, height, width, channels)
                # offsets = np.expand_dims(offsets, axis=0)  # Shape (1, n)

                # Predict action probabilities and estimated future rewards
                action_probs, critic_value = model([x_con, y_con])

                critic_value_history.append(critic_value[0, 0])

                # Sample action from action probability distributions
                if np.random.rand() < epsilon:
                    action = 0
                    while action//(env.n*n_moves) == (action//n_moves)%env.n:
                        action = np.random.choice(env.n * env.n * n_moves)  # Random action
                else:
                    action = np.random.choice(env.n * env.n * n_moves, p=action_probs.numpy()[0])  # Best action
                    #action = np.argmax(action_probs.numpy()[0])
                action_probs_history.append(keras.ops.log(action_probs[0, action] + eps))

                first_choice, second_choice, move = (
                    action//(env.n*n_moves),
                    (action//n_moves)%env.n,
                    action%n_moves
                )
                # Apply the sampled action in our environment
                _, reward, done, _= env.think_step((first_choice, second_choice, move))
                max_reward_seen = max(max_reward_seen, reward)
                max_episode_reward = max(max_episode_reward, reward)
                rewards.append(reward)

                #if episode_count % 100 == 0:
                #print(rewards_history)
                #print(f"action taken (fc: {first_choice}, sc: {second_choice}, m: {move})")
                #env.fpp.visualize()

                if done:
                    break

            episode_reward += max_episode_reward/batch_size
            # Calculate expected value from rewards
            # - At each timestep what was the total reward received after that timestep
            # - Rewards in the past are discounted by multiplying them with gamma
            # - These are the labels for our critic
            returns = []
            max_val = -np.inf
            for r in rewards[::-1]:
                max_val = max(max_val, r)
                returns.insert(0, max_val)

            returns_history.extend(returns)

        # Update running reward
        running_reward = 0.05 * episode_reward + (1 - 0.05) * running_reward

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

        if episode_count % 1 == 0:
            # Log details
            template = ("Best objective: {:.2f}, "
                        "actual objective: {:.2f}, "
                        "running reward: {:.2f}, "
                        "critic error: {:.2f}, "
                        "actors error: {:.2f}, "
                        "episode reward: {:.2f} "
                        "epsilon: {:.2f}, "
                        "and number of moves {}  at episode {}")
            print(template.format(
                abs(env.best_obj),
                env.current_fpp.get_current_sp_objective(),
                running_reward,
                np.mean(critic_losses),
                np.mean(actor_losses),
                np.mean(returns_history),
                epsilon,
                len(critic_losses)/batch_size,
                episode_count)
            )

        # Clear the loss and reward history
        action_probs_history.clear()
        critic_value_history.clear()
        returns_history.clear()
        epsilon = epsilon*epsilon_decay

env.close()