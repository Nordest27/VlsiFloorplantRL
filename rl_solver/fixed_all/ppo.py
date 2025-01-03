from collections import deque
import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Input, Dense, Flatten, Reshape, Conv2DTranspose, Conv2D, MaxPooling2D, Lambda, Concatenate
from tensorflow.keras.applications import EfficientNetB7
from tensorflow.keras import Model
import random

from floorplant_gym_env import FloorPlantEnv, save_floorplan_image

import tensorflow as tf

def edge_loss(y_true, y_pred):
    sobel_x_pred = tf.image.sobel_edges(y_pred)[..., 0]
    sobel_y_pred = tf.image.sobel_edges(y_pred)[..., 1]
    sobel_x_true = tf.image.sobel_edges(y_true)[..., 0]
    sobel_y_true = tf.image.sobel_edges(y_true)[..., 1]

    # Compute the gradient magnitude for both true and predicted images
    edge_pred = tf.sqrt(tf.square(sobel_x_pred) + tf.square(sobel_y_pred) + 1e-7)
    edge_true = tf.sqrt(tf.square(sobel_x_true) + tf.square(sobel_y_true) + 1e-7)

    # Return the mean squared error between the true and predicted edges
    return tf.reduce_mean(tf.square(edge_true - edge_pred))

def total_variation_loss(y_true, y_pred):
    return tf.reduce_mean( (tf.image.total_variation(y_true) - tf.image.total_variation(y_pred))**2 )

# print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))
# ret
# Hyperparameters

gamma = 0.99
clip_epsilon = 0.2
entropy_coefficient = .01
critic_coefficient = 0.5
learning_rate = 1e-3
autoencoder_minibatch_size = 32
minibatch_size = 64
sampling_batch = 64

# Environment setup
env = FloorPlantEnv(16)
n_moves = 9

env.max_steps = 8

batch_size = minibatch_size*4
ppo_epochs = batch_size//minibatch_size

obs_shape = (int(env.max_offset), int(env.max_offset), 3)

#weighted_connections_size = env.n * (env.n - 1) // 2
# Define the actor-critic model
#effnet_base = EfficientNetB7(include_top=False, weights=None, input_shape=obs_shape)
#effnet_base.trainable = True
# # Model architecture
# actor_input_layer = Input(shape=obs_shape)
# effnet_output = effnet_base(actor_input_layer)
# effnet_output_flattened = Flatten()(effnet_output)
# Latent space (bottleneck)
#latent_dim = 512  # Optional size for latent representation
#latent = Dense(latent_dim, activation='relu')(effnet_output_flattened)

# Shared Encoder
encoder_input = Input(shape=obs_shape)
x = Conv2D(64, (5, 5), activation='relu', padding='same')(encoder_input)
x = MaxPooling2D((2, 2), padding='same')(x)
x = Conv2D(128, (5, 5), activation='relu', padding='same')(x)
x = MaxPooling2D((2, 2), padding='same')(x)
x = Flatten()(x)
latent = Dense(env.n*8, activation='relu')(x)

actor_layer = Dense(256, activation='relu')(latent)
actor_layer = Dense(128, activation='relu')(actor_layer)
wea = Dense(env.n, activation="softmax", name="wfa")(actor_layer)
# wsa = Dense(env.n//2, activation="softmax", name="wsa")(actor_layer)
wma = Dense(n_moves, activation="softmax", name="wma")(actor_layer)

#
# # Clipping logits before softmax
# def clip_logits(logits):
#     return tf.clip_by_value(logits, clip_value_min=-5.0, clip_value_max=5.0)
#
# # Actions with logits clipping
# wea_logits = Dense(env.n, name="wea_logits")(actor_layer)
# wea_clipped = Lambda(clip_logits)(wea_logits)
# wea = Lambda(tf.nn.softmax, name="wea")(wea_clipped)
#
# # wsa_logits = Dense(env.n//2, name="wsa_logits")(Concatenate()([actor_layer, wfa_logits]))
# # wsa_clipped = Lambda(clip_logits)(wsa_logits)
# # wsa = Lambda(tf.nn.softmax, name="wsa")(wsa_clipped)
#
# wma_logits = Dense(n_moves, name="wma_logits")(Concatenate()([actor_layer, wea_logits]))
# wma_clipped = Lambda(clip_logits)(wma_logits)
# wma = Lambda(tf.nn.softmax, name="wma")(wma_clipped)


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

# # Decoder input
# decoder2_input = Dense((obs_shape[0] // 4) * (obs_shape[1] // 4) * 128, activation='relu')(latent)
# decoder2_input_reshaped = Reshape((obs_shape[0] // 4, obs_shape[1] // 4, 128))(decoder2_input)
#
# # Upsampling to match the original input shape
# decoder2_output = Conv2DTranspose(64, (5, 5), strides=(2, 2), padding='same', activation='relu')(decoder2_input_reshaped)
#
# # Final layer to output image with 3 channels (RGB)
# decoder2_output = Conv2DTranspose(3, (5, 5), strides=(2, 2), padding='same', activation='relu')(decoder2_output)
#
model = Model(inputs=encoder_input, outputs=[wea, wma, critic, decoder_output])
optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
model.summary()
zero = tf.constant(0)

targets = []
inps = []
for _ in range(0):
    env.fpp.shuffle_sp()
    env.steps = random.randint(0, env.max_steps)
    #target = env.draw(env.fpp)
#     for index, v in enumerate(env.fpp.widths()+env.fpp.heights()):
#         #target[index] += v/2
#         target[index] /= env.n
    #print(x_target)
    #print(y_target)
    #env.fpp.visualize()
    #targets.append(target)
    inps.append(env.draw(env.fpp).numpy())

for i in range(0):
    # Get old values for value loss
    aux_indices = np.random.choice(len(inps), autoencoder_minibatch_size, replace=False)
    aux_targets = []
    aux_inps = []
    for idx in aux_indices:
        aux_inps.append(inps[idx])
    with tf.GradientTape() as tape:
        aux_inps = tf.convert_to_tensor(aux_inps, dtype=tf.float32)
        values = model(aux_inps)[4]
        loss = (aux_inps - values)**2 #+ 0.001*edge_loss(aux_inps, values) #+ 0.01*total_variation_loss(aux_inps, values)
    if i%10 == 0:
        print(f"Iter {i}: {tf.reduce_mean(loss).numpy()}")
    grads = tape.gradient((zero, zero, zero, loss), model.trainable_variables)
    optimizer.apply_gradients(zip(grads, model.trainable_variables))
    if i%100 == 0:
        save_floorplan_image(aux_inps[0], "visualizations/current_model_input.png")
        save_floorplan_image(values[0], "visualizations/current_model_output.png")

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
        values = tf.squeeze(model(states)[2], axis=-1)
        next_values = tf.squeeze(model(next_states)[2], axis=-1)
        target_values = rewards + gamma * next_values * (1 - dones)
        #target_values = rewards
        advantages = target_values - values
        critic_loss = tf.reduce_mean((values - target_values) ** 2)
#         tf.debugging.check_numerics(critic_loss, "Loss contains NaN or Inf")

#     advantages = (advantages - tf.reduce_mean(advantages)) / (tf.math.reduce_std(advantages) + 1e-8)

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
#             mb_rewards = tf.gather(rewards, batch_indices)
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
#         with tf.GradientTape() as tape:
#             prediction_values = model(mb_states)[5]
#             loss = (mb_next_states - prediction_values)**2 #+ 0.001*edge_loss(aux_inps, values) #+ 0.01*total_variation_loss(aux_inps, values)
#
#         grads = tape.gradient((zero, zero, zero, zero, zero, loss), model.trainable_variables)
#         optimizer.apply_gradients(zip(grads, model.trainable_variables))
    print("State:" )
    save_floorplan_image(mb_states[0], "visualizations/current_model_input.png")
    save_floorplan_image(mb_next_states[0], "visualizations/current_model_next.png")
    save_floorplan_image(decoder_values[0], "visualizations/current_model_output.png")
#     save_floorplan_image(prediction_values[0], "visualizations/current_model_prediction.png")

    #for i in range(len(mb_states)):
    #    print(f"Comp index i: {mb_indices[0]}, j: {mb_indices[i]}, cmp: {tf.reduce_sum(mb_states[i]-mb_states[0])}")

    return kl_div_reached
    #return np.mean(actor_loss), critic_loss.numpy()


# Main training loop
def main():
    # Training parameters
    buffer = deque(maxlen=1000)
    num_episodes = 10000

    # Training loop
    running_reward = 0
    for episode in range(num_episodes):
        env.reset()
        sps_to_expand = deque(maxlen=sampling_batch)
        states_to_expand = deque(maxlen=sampling_batch)
        rand_sp_to_expand = deque(maxlen=sampling_batch)
        for i in range(sampling_batch):
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
        initial = True
        steps = 0
        episode_buffer = deque(maxlen=sampling_batch*env.max_steps)
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
                next_state = env.draw(env.fpp)
                if initial:
                    print((first_choice, second_choice, move))
                    print(wep_dist[first_choice], wep_dist[second_choice], wmp_dist[move])
#                     if wfp_dist[first_choice] > 0.99 and wsp_dist[second_choice] > 0.99 and  wmp_dist[move] > 0.99:
#                         print("Very confident! permanently applying move:", (first_choice, second_choice, move))
#                         env.ini_fpp = env.fpp.copy()
#                         env.rand_ini_fpp = env.rand_fpp.copy()

                if episode % 10 == 8 and sp_i == 0:
                    print(first_choice, second_choice, move)

                episode_buffer.append(
                    [states[sp_i], action, reward, next_state, done,
                     [tf.math.log(wep_dist[first_choice]),
                      tf.math.log(wep_dist[second_choice]),
                      tf.math.log(wmp_dist[move])]
                     ]
                )
                episode_reward += reward

#                 if str((env.fpp.x(), env.fpp.y())) not in expanded_states: # and reward > 0:
                sps_to_expand.append((env.fpp.x(), env.fpp.y()))
                states_to_expand.append(next_state)
#                 if str((env.rand_fpp.x(), env.rand_fpp.y())) not in rand_expanded_states: # and (reward > 0 or random.random() > 0.75):
                rand_sp_to_expand.append((env.rand_fpp.x(), env.rand_fpp.y()))
                initial = False
            steps += 1

        #
        #                 if episode % 100 == 0:
        #                     print("Rewards:")
        #                     for offset in range(sampling_batch):
        #                         text = f"sample run {offset}: "
        #                         for i in range(offset, len(buffer), sampling_batch):
        #                             text += f"{int(10*rewards[i])},"
        #                         print(text)
        #
        """
        for offset in range(sampling_batch):
            for i in reversed(range(offset, len(episode_buffer), sampling_batch)):
                if i - sampling_batch >= 0:
                    episode_buffer[i-sampling_batch][2] += gamma*episode_buffer[i][2]
        """
        buffer.extend(episode_buffer)
        #
        #                 if episode % 100 == 0:
        #                     print("Rewards after:")
        #                     for offset in range(sampling_batch):
        #                         text = f"sample run {offset}: "
        #                         for i in range(offset, len(buffer), sampling_batch):
        #                             text += f"{int(10*rewards[i])},"
        #                         print(text)

        if len(buffer) >= batch_size:
            print("Updating network...")
            states, actions, rewards, next_states, dones, old_probs = zip(*random.sample(buffer, batch_size))
            kl_stop = False
            it_up = 0
            while not kl_stop and it_up <= 10:
                kl_stop = update_model(
                    tf.convert_to_tensor(np.array(states), dtype=tf.float32),
                    [tf.one_hot(a, dim) for a, dim in zip(zip(*actions), [env.n, env.n, n_moves])],
                    tf.convert_to_tensor(rewards, dtype=tf.float32),
                    tf.convert_to_tensor(np.array(next_states), dtype=tf.float32),
                    tf.convert_to_tensor(dones, dtype=tf.float32),
                    [tf.convert_to_tensor(p, dtype=tf.float32) for p in zip(*old_probs)],
                )
                it_up += 1
            buffer.clear()
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
