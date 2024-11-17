# using example https://keras.io/examples/rl/actor_critic_cartpole/
import numpy as np
import os
#os.environ["KERAS_BACKEND"] = "tensorflow"
from keras import backend
from numpy.ma.core import true_divide
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


env = FloorPlantEnv(25)
input_layer = keras.layers.Input((env.n + env.n,))

hidden_layer = keras.layers.Embedding(env.n+1, 4)(input_layer)
hidden_layer = keras.layers.Flatten()(hidden_layer)
hidden_layer = keras.layers.Dense(100, activation="relu")(hidden_layer)

actor_hidden_layer = keras.layers.Dense(100, activation="tanh")(hidden_layer)
actor = keras.layers.Dense(env.n + env.n, activation='softmax')(actor_hidden_layer)

critic_hidden_layer = keras.layers.LeakyReLU(negative_slope=0.1)(keras.layers.Dense(50)(hidden_layer))
critic = keras.layers.Dense(1)(critic_hidden_layer)

model = keras.Model(inputs=input_layer, outputs=[actor, critic])

# Define optimizer and loss functions
optimizer = keras.optimizers.Adam(learning_rate=1e-3)
# model.summary()

# Main training loop
num_episodes = 1000
gamma = 0.99
eps = np.finfo(np.float32).eps.item()
action_probs_history = []
critic_value_history = []
returns_history = []
running_reward = 0
episode_count = 0

temperature = 10 # 1.0 is neutral; lower values for exploitation, higher for exploration
min_temperature = 0.1
decay_rate = 1

batch_size = 1

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
            env.visualize_sa_solution()
            env.best_fpp.visualize()


        for batch_i in range(batch_size):
            env.reset()

            rewards = []
            done = False
            while not done: #range(2+int(np.log2(1+episode))):  # Limit the number of time steps

                # Predict action probabilities and estimated future rewards
                # from environment state
                action_probs, critic_value = model(
                    keras.ops.expand_dims(
                        keras.ops.convert_to_tensor(env.get_fpp_input()), 0
                    )
                )
                critic_value_history.append(critic_value[0, 0])

                # Sample action from action probability distributions
                temp_action_probs = apply_temperature(action_probs, temperature)[0]

                # Mask invalid moves
                for v in env.observation[0]:
                    if v != env.n:
                        temp_action_probs[v] = 0
                for v in env.observation[1]:
                    if v != env.n:
                        temp_action_probs[env.n + v] = 0
                #print(temp_action_probs)
                
                temp_action_probs = temp_action_probs / sum(temp_action_probs)

                action = np.random.choice(env.n + env.n, p=temp_action_probs)

                action_probs_history.append(keras.ops.log(temp_action_probs[action]))

                value = action % env.n
                is_y = action >= env.n

                # Apply the sampled action in our environment
                reward, done= env.action(value, is_y)

                rewards.append(reward)
                episode_reward += reward/batch_size

                #if episode_count % 100 == 0:
                #print(rewards_history)
                #print(f"action taken (fc: {first_choice}, sc: {second_choice}, m: {move})")
                #env.fpp.visualize()

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

        # Calculating loss values to update our network
        # print(returns_history)
        # print(critic_value_history)
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

            actor_losses.append(-log_prob * diff * int(temperature < 2.5))

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
        template = (
            "temp: {:.2f}, "
            "best objective: {:.2f}, "
            "running reward: {:.2f}, "
            "critic error: {:.2f}, "
            "actors error: {:.2f}, "
            "episode reward: {:.2f} "
            "and number of moves {}  at episode {}")
        print(template.format(
            temperature,
            abs(env.best_obj),
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

        # Update temperature over episodes
        temperature = max(min_temperature, temperature * decay_rate)

how_many_samples = 10000
best_fpp = None
best_value = -np.inf
for sample in range(how_many_samples):
    env.reset()
    action_probs, critic_value = model(
        keras.ops.expand_dims(
            keras.ops.convert_to_tensor(env.get_fpp_input()), 0
        )
    )
    if critic_value > best_value:
        print("Found better seed:", critic_value)
        best_value = critic_value
        best_fpp = env.fpp.copy()

print("Using simulated annealing...")
best_fpp.apply_simulated_annealing(100, 1.0 - 1e-5)
obj = -best_fpp.get_current_sp_objective()

print("Objective found: ", obj)
best_fpp.visualize()


env.close()