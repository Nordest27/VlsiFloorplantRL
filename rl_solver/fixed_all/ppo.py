from collections import deque
import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Input, Dense, Flatten
from tensorflow.keras.applications import EfficientNetB0
from tensorflow.keras import Model
import random

from floorplant_gym_env import FloorPlantEnv

# Hyperparameters
gamma = 0.99
clip_epsilon = 0.2
entropy_coefficient = 0.01
critic_coefficient = 0.5
learning_rate = 1e-3
ppo_epochs = 1
minibatch_size = 64
batch_size = 512

# Environment setup
env = FloorPlantEnv(16)
n_moves = 9
env.max_steps = batch_size
obs_shape = (max(env.n, 32), max(env.n, 32), 3)
weighted_connections_size = env.n * (env.n - 1) // 2

# Define the actor-critic model
effnet_base = EfficientNetB0(include_top=False, input_shape=obs_shape)
effnet_base.trainable = True

# Model architecture
actor_input_layer = Input(shape=obs_shape)
effnet_output = effnet_base(actor_input_layer)
effnet_output_flattened = Flatten()(effnet_output)

wfa = Dense(env.n, activation="softmax", name="wfa")(effnet_output_flattened)
wsa = Dense(env.n, activation="softmax", name="wsa")(effnet_output_flattened)
wma = Dense(n_moves, activation="softmax", name="wma")(effnet_output_flattened)
critic = Dense(1, name="critic")(effnet_output_flattened)
coords = Dense(env.n+env.n, name="coords")(effnet_output_flattened)

model = Model(inputs=actor_input_layer, outputs=[wfa, wsa, wma, critic])
optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
model.summary()

def ppo_loss(old_log_probs, log_probs, advantages):
    ratios = tf.exp(log_probs - old_log_probs)
    clipped_ratios = tf.clip_by_value(ratios, 1 - clip_epsilon, 1 + clip_epsilon)
    surrogate1 = ratios * advantages
    surrogate2 = clipped_ratios * advantages
    return -tf.reduce_mean(tf.minimum(surrogate1, surrogate2))

# Define the target KL divergence threshold
tgt_KL = 0.01  # You can tune this value
max_KL = 1.5 * tgt_KL  # The stopping criterion

# Update model using PPO with KL divergence stopping criterion
def update_model(states, actions, rewards, next_states, dones, old_probs):
    zero = tf.constant(0)

    # Get old values for value loss
    with tf.GradientTape() as tape:
        values = tf.squeeze(model(states)[3], axis=-1)
        next_values = tf.squeeze(model(next_states)[3], axis=-1)
        target_values = rewards + gamma * next_values
        #target_values = rewards
        advantages = target_values - values
        critic_loss = tf.reduce_mean((values - target_values) ** 2)

    advantages = (advantages - tf.reduce_mean(advantages)) / (tf.math.reduce_std(advantages) + 1e-8)

    grads = tape.gradient((zero, zero, zero, critic_loss), model.trainable_variables)
    optimizer.apply_gradients(zip(grads, model.trainable_variables))

    # Update actor using PPO with KL divergence stopping
    for _ in range(ppo_epochs):
        kl_wfp = tf.constant(0.0)
        kl_wsp = tf.constant(0.0)
        kl_wmp = tf.constant(0.0)
        indices = np.arange(len(states))
        np.random.shuffle(indices)
        for start in range(0, len(states), minibatch_size):
            end = start + minibatch_size
            batch_indices = indices[start:end]

            mb_states = tf.gather(states, batch_indices)
            mb_actions = [tf.gather(a, batch_indices) for a in actions]

            mb_advantages = tf.gather(advantages, batch_indices)
            mb_old_probs = [tf.gather(p, batch_indices) for p in old_probs]

            with tf.GradientTape() as tape:
                # Get new log probabilities for actions
                wfp, wsp, wmp, _ = model(mb_states)
                log_probs_wfp = tf.reduce_sum(mb_actions[0] * tf.math.log(wfp + 1e-8), axis=1)
                log_probs_wsp = tf.reduce_sum(mb_actions[1] * tf.math.log(wsp + 1e-8), axis=1)
                log_probs_wmp = tf.reduce_sum(mb_actions[2] * tf.math.log(wmp + 1e-8), axis=1)

                # PPO loss for each action space
                kl_wfp += tf.reduce_mean(abs(mb_old_probs[0]-log_probs_wfp))/(batch_size/ppo_epochs)
                actor_loss_wfp = ppo_loss(mb_old_probs[0], log_probs_wfp, mb_advantages)
                kl_wsp += tf.reduce_mean(abs(mb_old_probs[1]-log_probs_wsp))/(batch_size/ppo_epochs)
                actor_loss_wsp = ppo_loss(mb_old_probs[1], log_probs_wsp, mb_advantages)
                kl_wmp += tf.reduce_mean(abs(mb_old_probs[2]-log_probs_wmp))/(batch_size/ppo_epochs)
                actor_loss_wmp = ppo_loss(mb_old_probs[2], log_probs_wmp, mb_advantages)

                # Add entropy regularization
                actor_loss_wfp += entropy_coefficient * tf.reduce_mean(wfp * tf.math.log(wfp + 1e-8))
                actor_loss_wsp += entropy_coefficient * tf.reduce_mean(wsp * tf.math.log(wsp + 1e-8))
                actor_loss_wmp += entropy_coefficient * tf.reduce_mean(wmp * tf.math.log(wmp + 1e-8))

                # Combine all losses
                actor_loss = (
                    actor_loss_wfp,
                    actor_loss_wsp,
                    actor_loss_wmp,
                )

            # Backpropagate loss
            grads = tape.gradient((*actor_loss, tf.constant(0)), model.trainable_variables)
            optimizer.apply_gradients(zip(grads, model.trainable_variables))

        # If KL divergence exceeds threshold, stop the update
        if kl_wfp > max_KL or kl_wsp > max_KL or kl_wmp > max_KL:
            print(f"KL Divergence exceeded threshold, stopping update: KL_wfp={kl_wfp}, KL_wsp={kl_wsp}, KL_wmp={kl_wmp}")
            return True
    return False
    #return np.mean(actor_loss), critic_loss.numpy()


# Main training loop
def main():
    # Training parameters
    buffer = deque(maxlen=batch_size)
    num_episodes = 10000

    # Training loop
    running_reward = 0
    for episode in range(num_episodes):

        sps_to_expand = []
        states_to_expand = []
        rand_sp_to_expand = []

        for i in range(1):
            env.reset()
            state = env.draw(env.fpp)
            #state = np.ravel(np.append(env.flattened_observation()[0], env.flattened_observation()[1]))
            sps_to_expand.append((env.fpp.x(), env.fpp.y()))
            states_to_expand.append(state)
            rand_sp_to_expand.append((env.rand_fpp.x(), env.rand_fpp.y()))


        expanded_states = set()
        rand_expanded_states = set()

        if episode % 10 == 9:
            print("Simulated Annealing solution:", env.sa_fpp.get_current_sp_objective())
            env.sa_fpp.visualize()
            env.best_fpp.visualize()

        episode_reward = 0
        done = False
        while not done:
            sp_indices = np.random.choice(len(states_to_expand), min(minibatch_size, len(states_to_expand)), replace=False)
            states = [states_to_expand[i] for i in sp_indices]

            inp = tf.convert_to_tensor(states, dtype=tf.float32)
            lwfp, lwsp, lwmp, _ = model(inp)

            for bi in range(len(sp_indices)):
                sp_i = sp_indices[bi]
                env.fpp.set_sp(*sps_to_expand[sp_i])
                env.rand_fpp.set_sp(*rand_sp_to_expand[random.randint(0, len(rand_sp_to_expand)-1)])
                expanded_states.add(str((env.fpp.x(), env.fpp.y())))
                rand_expanded_states.add(str((env.rand_fpp.x(), env.rand_fpp.y())))

                # Action sampling
                wfp_dist = lwfp[bi].numpy()
                wsp_dist = lwsp[bi].numpy()
                wmp_dist = lwmp[bi].numpy()

                # Sample actions
                first_choice = np.random.choice(env.n, p=wfp_dist)
                wsp_dist[first_choice] = 0
                second_choice = np.random.choice(env.n, p=wsp_dist / wsp_dist.sum())
                move = np.random.choice(n_moves, p=wmp_dist)

                action = (first_choice, second_choice, move)
                _, reward, done, _ = env.step(action)
                next_state = env.draw(env.fpp)

                if episode % 10 == 8 and bi == 0:
                    print(first_choice, second_choice, move)

                buffer.append(
                    (state, action, reward, next_state, done,
                     [tf.math.log(wfp_dist[first_choice]),
                      tf.math.log(wsp_dist[second_choice]),
                      tf.math.log(wmp_dist[move])]
                     )
                )
                next_state = env.draw(env.fpp)
                episode_reward += reward

                if str((env.fpp.x(), env.fpp.y())) not in expanded_states:# and reward > 0:
                    sps_to_expand.append((env.fpp.x(), env.fpp.y()))
                    states_to_expand.append(next_state)
                if str((env.rand_fpp.x(), env.rand_fpp.y())) not in rand_expanded_states: # and (reward > 0 or random.random() > 0.75):
                    rand_sp_to_expand.append((env.rand_fpp.x(), env.rand_fpp.y()))

            if len(buffer) >= batch_size:
                print("Updating network...")
                states, actions, rewards, next_states, dones, old_probs = zip(*buffer)
                kl_stop = update_model(
                    tf.convert_to_tensor(np.array(states), dtype=tf.float32),
                    [tf.one_hot(a, dim) for a, dim in zip(zip(*actions), [env.n, env.n, n_moves])],
                    tf.convert_to_tensor(rewards, dtype=tf.float32),
                    tf.convert_to_tensor(np.array(next_states), dtype=tf.float32),
                    tf.convert_to_tensor(dones, dtype=tf.float32),
                    [tf.convert_to_tensor(p, dtype=tf.float32) for p in zip(*old_probs)],
                )
                if kl_stop:
                    buffer.clear()

            template = (
                "Episode: {}, "
                "Rand best obj: {:.2f}, "
                "Best obj {:.2f}, "
                "Episode reward: {:.2f}, "
                "Episode final state: {:.2f}, "
                "last move: {}"
            )

            print(template.format(
                episode,
                env.rand_best_fpp.get_current_sp_objective(),
                env.best_obj,
                episode_reward,
                env.fpp.get_current_sp_objective(),
                (first_choice, second_choice, move)
            ))

if __name__ == "__main__":
    main()
