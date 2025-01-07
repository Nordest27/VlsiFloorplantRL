from collections import deque
import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Input, Dense, Flatten, Reshape, Conv2DTranspose, Conv2D, MaxPooling2D, Lambda, Concatenate
from tensorflow.keras.applications import EfficientNetB7
from tensorflow.keras import Model
import random

from floorplant_gym_env import FloorPlantEnv, save_floorplan_image

import tensorflow as tf

# Hyperparameters
gamma = 0.99
clip_epsilon = 0.2
entropy_coefficient = .04
critic_coefficient = 0.5
learning_rate = 1e-3
autoencoder_minibatch_size = 32
minibatch_size = 64
sampling_batch = 16
batch_size = minibatch_size*4
ppo_epochs = batch_size//minibatch_size
n_moves = 3

def create_model(n: int):
    # Environment setup
    env = FloorPlantEnv(n)
    obs_shape = (int(env.max_offset), int(env.max_offset), 3)

    env.max_steps = 16

    # Shared Encoder
    encoder_input = Input(shape=obs_shape)
    x = Conv2D(64, (5, 5), activation='relu', padding='same')(encoder_input)
    x = MaxPooling2D((2, 2), padding='same')(x)
    x = Conv2D(128, (5, 5), activation='relu', padding='same')(x)
    x = MaxPooling2D((2, 2), padding='same')(x)
    x = Flatten()(x)
    latent = Dense(env.n*12, activation='relu')(x)

    actor_layer = Dense(256, activation='relu')(latent)
    actor_layer = Dense(128, activation='relu')(actor_layer)
    wea = Dense(env.n, activation="softmax", name="wfa")(actor_layer)
    wma = Dense(n_moves, activation="softmax", name="wma")(actor_layer)

    critic_layer = Dense(256, activation='relu')(latent)
    critic_layer = Dense(128, activation='relu')(critic_layer)
    critic = Dense(1, name="critic")(critic_layer)

    # Decoder input
    decoder_input = Dense((obs_shape[0] // 4) * (obs_shape[1] // 4) * 128, activation='relu')(latent)
    decoder_input_reshaped = Reshape((obs_shape[0] // 4, obs_shape[1] // 4, 128))(decoder_input)

    # Upsampling to match the original input shape
    decoder_output = Conv2DTranspose(64, (5, 5), strides=(2, 2), padding='same', activation='relu')(decoder_input_reshaped)

    # Final layer to output image with 3 channels (RGB)
    decoder_output = Conv2DTranspose(3, (5, 5), strides=(2, 2), padding='same', activation='relu')(decoder_output)

    # Ensure output shape matches the input shape
    assert decoder_output.shape[1:] == obs_shape, f"Decoder output shape {decoder_output.shape[1:]} does not match input shape {obs_shape}."

    model = Model(inputs=encoder_input, outputs=[wea, wma, critic, decoder_output])
    optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
    model.summary()
    zero = tf.constant(0)
    return env, model, optimizer


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
def update_model(model, optimizer, states, actions, rewards, next_states, dones, old_probs):
    zero = tf.constant(0)

    # Get old values for value loss
    with tf.GradientTape() as tape:
        values = tf.squeeze(model(states)[2], axis=-1)
        next_values = tf.squeeze(model(next_states)[2], axis=-1)
        target_values = rewards + gamma * next_values * (1 - dones)
        advantages = target_values - values
        critic_loss = tf.reduce_mean((values - target_values) ** 2)

#     print(f"Critic mean: {tf.reduce_mean(values)}, Critic error: {tf.reduce_mean(advantages)}")
#
    grads = tape.gradient((zero, zero, critic_loss, zero), model.trainable_variables)
    optimizer.apply_gradients(zip(grads, model.trainable_variables))

    kl_div_reached = False
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
                wep, wmp, _, _ = model(mb_states)
                log_probs_wfp = tf.reduce_sum(mb_actions[0] * tf.math.log(wep + 1e-8), axis=1)
                log_probs_wsp = tf.reduce_sum(mb_actions[1] * tf.math.log(wep + 1e-8), axis=1)
                log_probs_wmp = tf.reduce_sum(mb_actions[2] * tf.math.log(wmp + 1e-8), axis=1)

                # PPO loss for each action space
                kl_wfp += tf.reduce_mean(abs(mb_old_probs[0]-log_probs_wfp))/(batch_size/ppo_epochs)
                actor_loss_wfp = ppo_loss(mb_old_probs[0], log_probs_wfp, mb_advantages)
                kl_wsp += tf.reduce_mean(abs(mb_old_probs[1]-log_probs_wsp))/(batch_size/ppo_epochs)
                actor_loss_wsp = ppo_loss(mb_old_probs[1], log_probs_wsp, mb_advantages)
                kl_wmp += tf.reduce_mean(abs(mb_old_probs[2]-log_probs_wmp))/(batch_size/ppo_epochs)
                actor_loss_wmp = ppo_loss(mb_old_probs[2], log_probs_wmp, mb_advantages)

                # Add entropy regularization
                actor_loss_wfp += entropy_coefficient * tf.reduce_mean(wep * tf.math.log(wep + 1e-8))
                actor_loss_wsp += entropy_coefficient * tf.reduce_mean(wep * tf.math.log(wep + 1e-8))
                actor_loss_wmp += entropy_coefficient * tf.reduce_mean(wmp * tf.math.log(wmp + 1e-8))

                # Combine all losses
                actor_loss = (
                    (actor_loss_wfp + actor_loss_wsp)/2,
                    actor_loss_wmp,
                )
                tf.debugging.check_numerics(actor_loss, "Loss contains NaN or Inf")

            # Backpropagate loss
            grads = tape.gradient((*actor_loss, zero, zero), model.trainable_variables)
            optimizer.apply_gradients(zip(grads, model.trainable_variables))

        # If KL divergence exceeds threshold, stop the update
        if kl_wfp > max_KL or kl_wsp > max_KL or kl_wmp > max_KL:
            print(f"KL Divergence exceeded threshold, stopping update: KL_wfp={kl_wfp}, KL_wsp={kl_wsp}, KL_wmp={kl_wmp}")
            kl_div_reached = True
            break

    indices = np.arange(len(states))
    np.random.shuffle(indices)
    print("First index:", indices[0])
    for start in range(0, len(states), minibatch_size):
        end = start + minibatch_size
        batch_indices = indices[start:end]

        mb_states = tf.gather(states, batch_indices)
        with tf.GradientTape() as tape:
            decoder_values = model(mb_states)[3]
            loss = (mb_states - decoder_values)**2 #+ 0.001*edge_loss(aux_inps, values) #+ 0.01*total_variation_loss(aux_inps, values)
            tf.debugging.check_numerics(loss, "Loss contains NaN or Inf")

        grads = tape.gradient((zero, zero, zero, loss), model.trainable_variables)
        optimizer.apply_gradients(zip(grads, model.trainable_variables))

        mb_next_states = tf.gather(next_states, batch_indices)

    print("State:" )
    save_floorplan_image(mb_states[0], "visualizations/current_model_input.png")
    save_floorplan_image(mb_next_states[0], "visualizations/current_model_next.png")
    save_floorplan_image(decoder_values[0], "visualizations/current_model_output.png")

    return kl_div_reached


# Main training loop
def main(n: int):
    # Training parameters
    env, model, optimizer = create_model(n)

    buffer = deque(maxlen=batch_size)
    num_episodes = 100

    # Training loop
    running_reward = 0
    f = open("results.csv", "w")
    rand_f = open("rand_results.csv", "w")
    f.write("Episode,Value\n")
    rand_f.write("Episode,Value\n")
    for episode in range(num_episodes):
        env.reset()
        sps_to_expand = deque(maxlen=sampling_batch)
        states_to_expand = deque(maxlen=sampling_batch)
        rand_sp_to_expand = deque(maxlen=sampling_batch)
        for i in range(sampling_batch):
            env.reset()
            state = env.draw(env.fpp)
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
        initial = True
        steps = 0
        while not done:
            sp_indices = np.random.choice(len(states_to_expand), min(sampling_batch, len(states_to_expand)), replace=False)
            sp_indices = range(sampling_batch)
            states = [states_to_expand[sp_i] for sp_i in sp_indices]
            sps = [sps_to_expand[sp_i] for sp_i in sp_indices]
            rand_sps = [rand_sp_to_expand[sp_i] for sp_i in sp_indices]

            inp = tf.convert_to_tensor(states, dtype=tf.float32)
            lwep, lwmp, _, _ = model(inp)

            for sp_i in range(len(states)):
                env.steps = steps
                env.fpp.set_sp(*sps[sp_i])
                assert tf.reduce_all(tf.math.equal(env.draw(env.fpp), states[sp_i]))
                env.rand_fpp.set_sp(*rand_sps[sp_i])
                expanded_states.add(str((env.fpp.x(), env.fpp.y())))
                rand_expanded_states.add(str((env.rand_fpp.x(), env.rand_fpp.y())))

                # Action sampling
                wep_dist = lwep[sp_i].numpy()
                wep_dist += 1e-8
                wep_dist /= wep_dist.sum()
                wmp_dist = lwmp[sp_i].numpy()
                wmp_dist += 1e-8
                wmp_dist /= wmp_dist.sum()

                # Sample actions
                first_choice, second_choice = np.random.choice(env.n, 2, replace=False, p=wep_dist)
                move = np.random.choice(n_moves, p=wmp_dist)

                action = (first_choice, second_choice, move)
                _, reward, done, _ = env.step(action)
                f.write(f"{episode},{env.fpp.get_current_sp_objective()}\n")
                rand_f.write(f"{episode},{env.rand_fpp.get_current_sp_objective()}\n")
                next_state = env.draw(env.fpp)
                if initial:
                    print((first_choice, second_choice, move))
                    print(wep_dist[first_choice], wep_dist[second_choice], wmp_dist[move])

                if episode % 10 == 8 and sp_i == 0:
                    print(first_choice, second_choice, move)

                buffer.append(
                    [states[sp_i], action, reward, next_state, done,
                     [tf.math.log(wep_dist[first_choice]),
                      tf.math.log(wep_dist[second_choice]),
                      tf.math.log(wmp_dist[move])]
                     ]
                )
                episode_reward += reward

                sps_to_expand.append((env.fpp.x(), env.fpp.y()))
                states_to_expand.append(next_state)
                rand_sp_to_expand.append((env.rand_fpp.x(), env.rand_fpp.y()))
                initial = False
            steps += 1

        print(len(buffer))
        if len(buffer) >= batch_size:
            print("Updating network...")
            states, actions, rewards, next_states, dones, old_probs = zip(*random.sample(buffer, batch_size))
            kl_stop = False
            it_up = 0
            while not kl_stop and it_up <= 10:
                kl_stop = update_model(
                    model, optimizer,
                    tf.convert_to_tensor(np.array(states), dtype=tf.float32),
                    [tf.one_hot(a, dim) for a, dim in zip(zip(*actions), [env.n, env.n, n_moves])],
                    tf.convert_to_tensor(rewards, dtype=tf.float32),
                    tf.convert_to_tensor(np.array(next_states), dtype=tf.float32),
                    tf.convert_to_tensor(dones, dtype=tf.float32),
                    [tf.convert_to_tensor(p, dtype=tf.float32) for p in zip(*old_probs)],
                )
                it_up += 1
            buffer.clear()

        save_floorplan_image(env.draw(env.best_fpp), "visualizations/current_model_best.png")
        if done:
            template = (
                "Episode: {}, "
                "Rand best obj: {:.2f}, "
                "Best obj {:.2f}, "
                "Episode reward: {:.2f}, "
                "Episode final state: {:.2f}, "
                "Rand Episode final state: {:.2f}, "
                "last move: {}"
            )

            print(template.format(
                episode,
                env.rand_best_fpp.get_current_sp_objective(),
                env.best_obj,
                episode_reward/sampling_batch,
                env.fpp.get_current_sp_objective(),
                env.rand_fpp.get_current_sp_objective(),
                (first_choice, second_choice, move)
            ))
if __name__ == "__main__":
    main()
