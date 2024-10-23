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
env = FloorPlantEnv(5)

# Define the actor and critic networks
connected_input_layer = keras.layers.Input((env.n, env.n, 1))
reals_input_layer = keras.layers.Input((env.n*2,))

# Convolutional layer for x_connected_input_layer
conv_layer = keras.layers.Conv2D(
    filters=8, kernel_size=3, padding='same', activation='relu'
)(connected_input_layer)
conv_layer = keras.layers.MaxPooling2D(pool_size=2)(conv_layer)
conv_layer = keras.layers.Flatten()(conv_layer)

#hidden_layer = keras.layers.LeakyReLU(negative_slope=0.1)(keras.layers.Dense(512)(hidden_layer))


# wfa = keras.layers.Dense(env.n, activation='softmax')(hidden_layer)
# wsa = keras.layers.Dense(env.n, activation='softmax')(hidden_layer)
# wma = keras.layers.Dense(10, activation='softmax')(hidden_layer)
actor_hidden_layer = keras.layers.Dense(512, activation="tanh")(conv_layer)

actor_x_outputs = [keras.layers.Dense(2, activation="softmax")(actor_hidden_layer)]
actor_y_outputs = [keras.layers.Dense(2, activation="softmax")(actor_hidden_layer)]
for i in range(3, env.n+1):
    actor_x_outputs.insert(0,
        keras.layers.Dense(i, activation="softmax")(
            keras.layers.Concatenate()([actor_hidden_layer, actor_x_outputs[0], actor_y_outputs[0]])
        )
    )
    actor_y_outputs.insert(0,
        keras.layers.Dense(i, activation="softmax")(
            keras.layers.Concatenate()([actor_hidden_layer, actor_x_outputs[0], actor_y_outputs[0]])
        )
    )

critic_hidden_layer = keras.layers.LeakyReLU(negative_slope=0.1)(keras.layers.Dense(512)(conv_layer))
critic = keras.layers.Dense(1)(critic_hidden_layer)

model = keras.Model(
    inputs=[reals_input_layer, connected_input_layer],
    outputs=[critic, *actor_x_outputs, *actor_y_outputs],
)
# model.summary()

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

batch_size = 20

env.reset()

for episode in range(num_episodes):
    episode_count += 1
    episode_reward = 0

    with GradientTape() as tape:
        if episode_count % 10 == 0:
            pass
            env.fpp.visualize_sa_solution()
            env.fpp.visualize()


        for batch_i in range(batch_size):
            env.reset()
            rewards = []

            # Assuming get_input returns offsets, x_con, and y_con as described
            dims, conn = env.get_input()

            # Convert to NumPy arrays
            dims = np.array(dims)  # Should be shape (n,) or (1, n) for a single batch
            conn = np.array(conn)  # Should be shape (height, width, channels)

            # Expand dimensions to add a batch dimension if necessary
            dims = np.expand_dims(dims, axis=0)  # Shape (1, n)
            conn = np.expand_dims(conn, axis=0)  # Shape (1, height, width, channels)

            critic_values, *action_probs_list = model([dims, conn])
            x_action = []
            x_choice = []
            x_chosen_action_probs = []
            x_indices = range(env.n)

            y_action = []
            y_choice = []
            y_chosen_action_probs = []
            y_indices = range(env.n)

            print(len(action_probs_list))
            for x_action_probs, y_action_probs in zip(
                action_probs_list[:len(action_probs_list)//2],
                action_probs_list[len(action_probs_list)//2:]   
            ):
                x_np_action_probs = x_action_probs.numpy()[0]
                x_choice.append(np.random.choice(len(x_np_action_probs), p=x_np_action_probs))
                x_action.append(x_indices[x_choice[-1]])
                x_chosen_action_probs.append(keras.ops.log(x_np_action_probs[x_choice[-1]]))
                x_indices = [i for i in x_indices if i != x_action[-1]]

                y_np_action_probs = y_action_probs.numpy()[0]
                y_choice.append(np.random.choice(len(y_np_action_probs), p=y_np_action_probs))
                y_action.append(y_indices[y_choice[-1]])
                y_chosen_action_probs.append(keras.ops.log(y_np_action_probs[x_choice[-1]]))
                y_indices = [i for i in y_indices if i != y_action[-1]]

            x_action.extend(x_indices)
            y_action.extend(y_indices)
            print(x_action)
            print(y_action)
            ret
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