import numpy as np
import tensorflow as tf
import gym

# Create the CartPole Environment
env = gym.make('CartPole-v1')

# Define the actor and critic networks
actor = tf.keras.Sequential([
    tf.keras.layers.Dense(32, activation='relu'),
    tf.keras.layers.Dense(env.action_space.n, activation='softmax')
])

critic = tf.keras.Sequential([
    tf.keras.layers.Dense(32, activation='relu'),
    tf.keras.layers.Dense(1)
])

# Define optimizer and loss functions
actor_optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)
critic_optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)

# Main training loop
num_episodes = 1000
gamma = 0.99

for episode in range(num_episodes):
    state = env.reset()
    episode_reward = 0

    with tf.GradientTape(persistent=True) as tape:
        for t in range(1, 10000):  # Limit the number of time steps
            # Choose an action using the actor
            action_probs = actor(np.array([state[0]]))
            action = np.random.choice(env.action_space.n, p=action_probs.numpy()[0])

            # Take the chosen action and observe the next state and reward
            next_state, reward, done, _, _ = env.step(action)

            # Compute the advantage
            state_value = critic(np.array([state[0]]))[0, 0]
            next_state_value = critic(np.array([next_state]))[0, 0]
            advantage = reward + gamma * next_state_value - state_value

            # Compute actor and critic losses
            actor_loss = -tf.math.log(action_probs[0, action]) * advantage
            critic_loss = tf.square(advantage)

            episode_reward += reward

            # Update actor and critic
            actor_gradients = tape.gradient(actor_loss, actor.trainable_variables)
            critic_gradients = tape.gradient(critic_loss, critic.trainable_variables)
            actor_optimizer.apply_gradients(zip(actor_gradients, actor.trainable_variables))
            critic_optimizer.apply_gradients(zip(critic_gradients, critic.trainable_variables))

            if done:
                break

    if episode % 10 == 0:
        print(f"Episode {episode}, Reward: {episode_reward}")

    env.close()
