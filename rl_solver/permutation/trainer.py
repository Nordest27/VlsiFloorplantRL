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

def apply_temperature(probs, temperature):
    """
    Apply temperature to smooth the action probabilities.
    Lower temperatures make the distribution sharper, while higher
    temperatures make it more uniform (encouraging exploration).
    """
    # Prevent division by zero
    if temperature == 0.0:
        return probs

    # Use temperature to scale log probabilities
    scaled_probs = np.log(probs + eps) / temperature
    exp_scaled = np.exp(scaled_probs)

    # Return normalized probabilities
    return exp_scaled / np.sum(exp_scaled)


# Create the CartPole Environment
env = FloorPlantEnv(15)

# Define the actor and critic networks
connected_input_layer = keras.layers.Input((env.n, env.n, 1))
reals_input_layer = keras.layers.Input((env.n*2,))

# Convolutional layer for x_connected_input_layer
actor_conv_layer = keras.layers.Conv2D(
    filters=8, kernel_size=3, padding='same', activation='relu'
)(connected_input_layer)
actor_conv_layer = keras.layers.MaxPooling2D(pool_size=2)(actor_conv_layer)
actor_conv_layer = keras.layers.Flatten()(actor_conv_layer)

actor_hidden_layer = keras.layers.Dense(512, activation="tanh")(actor_conv_layer)

actor_x_outputs = [keras.layers.Dense(2, activation="softmax")(actor_hidden_layer)]
actor_y_outputs = [keras.layers.Dense(2, activation="softmax")(actor_hidden_layer)]
for i in range(3, env.n+1):
    actor_x_outputs.insert(0,
        keras.layers.Dense(i, activation="softmax")(
            keras.layers.Concatenate()([actor_x_outputs[0]])
        )
    )
    actor_y_outputs.insert(0,
        keras.layers.Dense(i, activation="softmax")(
            keras.layers.Concatenate()([actor_x_outputs[0]])
        )
    )

critic_conv_layer = keras.layers.Conv2D(
    filters=8, kernel_size=3, padding='same', activation='relu'
)(connected_input_layer)
critic_conv_layer = keras.layers.MaxPooling2D(pool_size=2)(critic_conv_layer)
critic_conv_layer = keras.layers.Flatten()(critic_conv_layer)
critic_hidden_layer = keras.layers.LeakyReLU(negative_slope=0.1)(keras.layers.Dense(512)(critic_conv_layer))
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
x_action_probs_history = []
y_action_probs_history = []
critic_value_history = []
returns_history = []
running_reward = 0
episode_count = 0

temperature = 100 # 1.0 is neutral; lower values for exploitation, higher for exploration
min_temperature = 0.1
decay_rate = 0.9995

batch_size = 1

for episode in range(num_episodes):
    episode_count += 1
    episode_reward = 0
    # env.reset()
    with GradientTape() as tape:
        if episode_count % 100 == 0:
            pass
            env.fpp.visualize_sa_solution()
            env.fpp.visualize()


        for batch_i in range(batch_size):
            rewards = []

            # Assuming get_input returns offsets, x_con, and y_con as described
            dims, conn = env.get_input()

            # Convert to NumPy arrays
            dims = np.array(dims)  # Should be shape (n,) or (1, n) for a single batch
            conn = np.array(conn)  # Should be shape (height, width, channels)

            # Expand dimensions to add a batch dimension if necessary
            dims = np.expand_dims(dims, axis=0)  # Shape (1, n)
            conn = np.expand_dims(conn, axis=0)  # Shape (1, height, width, channels)

            critic_value, *action_probs_list = model([dims, conn])
            critic_value_history.append(critic_value)

            x_action = []
            x_choice = []
            x_chosen_action_probs = []
            x_indices = range(env.n)

            y_action = []
            y_choice = []
            y_chosen_action_probs = []
            y_indices = range(env.n)

            # Define a temperature parameter to control exploration
            for x_action_probs, y_action_probs in zip(
                    action_probs_list[:len(action_probs_list)//2],
                    action_probs_list[len(action_probs_list)//2:]
            ):
                x_np_action_probs = x_action_probs.numpy()[0]

                # Apply temperature to x_action_probs
                x_np_action_probs = apply_temperature(x_np_action_probs, temperature)
                x_choice.append(np.random.choice(len(x_np_action_probs), p=x_np_action_probs))
                x_action.append(x_indices[x_choice[-1]])
                x_chosen_action_probs.append(keras.ops.log(x_np_action_probs[x_choice[-1]]))
                x_indices = [i for i in x_indices if i != x_action[-1]]

                y_np_action_probs = y_action_probs.numpy()[0]

                # Apply temperature to y_action_probs
                y_np_action_probs = apply_temperature(y_np_action_probs, temperature)
                y_choice.append(np.random.choice(len(y_np_action_probs), p=y_np_action_probs))
                y_action.append(y_indices[y_choice[-1]])
                y_chosen_action_probs.append(keras.ops.log(y_np_action_probs[y_choice[-1]]))
                y_indices = [i for i in y_indices if i != y_action[-1]]


            x_action.extend(x_indices)
            y_action.extend(y_indices)

            x_action_probs_history.append(x_chosen_action_probs)
            y_action_probs_history.append(y_chosen_action_probs)

            reward = env.get_permutation(x_action, y_action)
            returns_history.append(reward)
            episode_reward += reward/batch_size

        # Update running reward
        running_reward = 0.05 * episode_reward + (1 - 0.05) * running_reward


        # Calculating loss values to update our network
        history = zip(x_action_probs_history, y_action_probs_history, critic_value_history, returns_history)
        actor_losses = []
        critic_losses = []
        for x_log_probs, y_log_probs, value, ret in history:
            # At this point in history, the critic estimated that we would get a
            # total reward = `value` in the future. We took an action with log probability
            # of `log_prob` and ended up receiving a total reward = `ret`.
            # The actor must be updated so that it predicts an action that leads to
            # high rewards (compared to critic's estimate) with high probability.
            diff = ret - value

            actor_losses.append(0)
            for log_prob in x_log_probs + y_log_probs:
                actor_losses[-1] += -log_prob * diff

            # The critic must be updated so that it predicts a better estimate of
            # the future rewards.
            critic_losses.append(diff ** 2)

        # Backpropagation
        loss_value = [
            0*sum(critic_losses)/len(critic_losses),
            sum(actor_losses)/(len(actor_losses)),
        ]
        #print("Loss value: ", loss_value)
        grads = tape.gradient(loss_value, model.trainable_variables)
        optimizer.apply_gradients(zip(grads, model.trainable_variables))


        if episode_count % 1 == 0:
            # Log details
            template = (
                "Temp: {:.2f}, "
                "best objective: {:.2f}, "
                "running reward: {:.2f}, "
                "critic error: {:.2f}, "
                "actors error: {:.2f}, "
                "episode reward: {:.2f} "
                "at episode {}"
            )
            print(template.format(
                temperature,
                abs(env.fpp.get_current_sp_objective()),
                running_reward,
                np.mean(critic_losses),
                np.mean(actor_losses),
                episode_reward,
                episode_count)
            )

        # Clear the loss and reward history
        x_action_probs_history.clear()
        y_action_probs_history.clear()
        critic_value_history.clear()
        returns_history.clear()
        # Update temperature over episodes
        temperature = max(min_temperature, temperature * decay_rate)

env.close()