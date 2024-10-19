import numpy as np
from tensorflow import keras, GradientTape, math, square
import gym
from floorplant_gym_env import FloorPlantEnv
import numpy as np


# Create the CartPole Environment
env = FloorPlantEnv(10)

# Define the actor and critic networks
input_layer = keras.layers.Input((env.n + env.n + env.n*4 + env.n*env.n,))
#hidden_layer = keras.layers.Dense(1000)(input_layer)
#hidden_layer = keras.layers.Dense(100)(keras.layers.LeakyReLU(negative_slope=0.05)(hidden_layer))
#hidden_layer_with_activation = keras.layers.LeakyReLU(negative_slope=0.05)(hidden_layer)
output_layer = keras.layers.concatenate([
    keras.layers.Dense(10, activation='softmax')(input_layer),
    keras.layers.Dense(10, activation='softmax')(input_layer),
    keras.layers.Dense(10, activation='softmax')(input_layer),
])
actor = keras.Model(input_layer, output_layer)

critic = keras.Sequential([
    keras.layers.Input((env.n + env.n + env.n*4 + env.n*env.n,)),
    #keras.layers.Dense(1000),
    #keras.layers.LeakyReLU(negative_slope=0.05),
    #keras.layers.Dense(100),
    #keras.layers.LeakyReLU(negative_slope=0.05),
    keras.layers.Dense(1, activation='sigmoid')
])

# Define optimizer and loss functions
actor_optimizer = keras.optimizers.Adam(learning_rate=0.000001)
critic_optimizer = keras.optimizers.Adam(learning_rate=0.000001)

# Main training loop
num_episodes = 1000
gamma = 1

env.reset()

for episode in range(num_episodes):
    print("New Env state")
    for v in env.observation:
        print(v)
    state = env.flattened_observation()
    env.fpp.visualize()
    episode_reward = 0

    with GradientTape(persistent=True) as tape:
        for t in range(1, 10000):  # Limit the number of time steps
            """
            action = (
                np.random.randint(0, env.n),
                np.random.randint(0, env.n),
                np.random.randint(0, 10)
            )
            while action[0] == action[1] :
                action = (
                    np.random.randint(0, env.n),
                    np.random.randint(0, env.n),
                    np.random.randint(0, 10)
               )

            next_state, reward, done,  _ = env.step(action)
            episode_reward += reward
            """

            # Choose an action using the actor
            action_probs = actor(np.array([env.flattened_observation()]))
            print(action_probs)
            first_choice = np.random.choice(env.n, p=action_probs.numpy()[0][:env.n])
            second_choice = np.random.choice(env.n, p=action_probs.numpy()[0][env.n:2*env.n])
            move = np.random.choice(10, p=action_probs.numpy()[0][env.n*2:])
            print(first_choice, second_choice, move)
            # Take the chosen action and observe the next state and reward
            _, reward, done, _= env.step((first_choice, second_choice, move))
            next_state = env.flattened_observation()

            # Compute the advantage
            state_value = critic(np.array([state]))[0, 0]
            next_state_value = critic(np.array([next_state]))[0, 0]
            advantage = reward + gamma * next_state_value - state_value

            # Compute actor and critic losses
            actor_loss = (
                -math.log(action_probs[0, first_choice])
                -math.log(action_probs[0, second_choice])
                -math.log(action_probs[0, move])
            ) * advantage
            critic_loss = square(advantage)

            episode_reward += reward

            # Update actor and critic
            actor_gradients = tape.gradient(actor_loss, actor.trainable_variables)
            critic_gradients = tape.gradient(critic_loss, critic.trainable_variables)
            actor_optimizer.apply_gradients(zip(actor_gradients, actor.trainable_variables))
            critic_optimizer.apply_gradients(zip(critic_gradients, critic.trainable_variables))

            env.render()
            if done:
                break

    if episode % 10 == 0:
        print(f"Episode {episode}, Reward: {episode_reward}")

env.close()