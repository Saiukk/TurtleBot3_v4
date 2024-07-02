import time

import torch
import torch as T
import torch.nn as nn
from collections import deque
import numpy as np
import torch.nn.functional as F
import gym
import random


class Network_cont(nn.Module):
    def __init__(self, input_shape, output_size, output_range = None, hiddenNodes=64):



        # Input -> 64 -> 64 -> output
        bound = 0.003
        super(Network_cont, self).__init__()
        self.input_layer = nn.Linear(in_features=input_shape, out_features=hiddenNodes)
        self.hidden = nn.Linear(in_features=hiddenNodes, out_features=hiddenNodes)
        self.hidden2 = nn.Linear(in_features=hiddenNodes, out_features=hiddenNodes)
        self.output_layer = nn.Linear(in_features=hiddenNodes,
                                      out_features=output_size)  # np.array(output_size).prod())
        nn.init.uniform_(self.output_layer.weight, -bound, bound)

        self.output_range = output_range


    def forward(self, x):
        # x = nn.functional.relu(self.input_layer(x))

        x = self.input_layer(x)
        x = nn.functional.relu(self.hidden(x))
        x = nn.functional.relu(self.hidden2(x))

        # with torch.no_grad():
        output = nn.functional.sigmoid(self.output_layer(x))
        if self.output_range is not None:
            output = (output * T.from_numpy((self.output_range[1] - self.output_range[0])) + T.from_numpy(self.output_range[0]))# in range [0,1]

        return output


class Network_critic(nn.Module):
    def __init__(self, input_shape, actor_action_shape):
        # Input -> 64 -> 64 -> output
        super(Network_critic, self).__init__()
        self.input_layer = nn.Linear(in_features=input_shape, out_features=32)
        self.input_action = nn.Linear(in_features=actor_action_shape, out_features=32)
        self.hidden_layer = nn.Linear(in_features=32, out_features=32)
        self.hidden_action = nn.Linear(in_features=32, out_features=32)

        self.hidden = nn.Linear(in_features=64, out_features=64)
        self.hidden2 = nn.Linear(in_features=64, out_features=64)
        self.output_layer = nn.Linear(in_features=64,
                                      out_features=1)  # np.array(output_size).prod())

    def forward(self, x, y):
        #print(x,'\n', y )
        x = self.input_layer(x)
        y = self.input_action(y)

        x = nn.functional.relu(self.hidden_layer(x))
        y = nn.functional.relu(self.hidden_action(y))

        concat = T.cat([x, y], dim=1)

        x = nn.functional.relu(self.hidden(concat))
        x = nn.functional.relu(self.hidden2(x))

        return self.output_layer(x)


class DDPG_PT:
    def __init__(self, env, verbose):
        self.env = env
        self.verbose = verbose
        self.device = T.device("cuda:0" if T.cuda.is_available() else "cpu")



        self.input_shape = self.env.observation_space.shape[0]
        self.action_shape = env.action_space.shape[0]
        print("OBS ->>", self.env.observation_space, "\n ACT-->", self.env.action_space)
        self.action_space = env.action_space.shape[0]
        print("OBS ->>", self.input_shape, "\n ACT-->", self.action_space)
        # output_rang = [env.action_space.low, env.action_space.high]

        self.actor = Network_cont(self.input_shape, self.action_space, [env.action_space.low, env.action_space.high]).to(self.device)
        self.critic = Network_critic(self.input_shape, self.action_shape).to(self.device)
        self.critic_target = Network_critic(self.input_shape, self.action_shape).to(self.device)

        # self.critic_target.set_weights(self.critic.get_weights())  # DA METTERE APPOSTO
        self.critic_target.load_state_dict(self.critic_target.state_dict())

        self.actor_optimizer = T.optim.Adam(self.actor.parameters())
        self.critic_optimizer = T.optim.Adam(self.critic.parameters())
        self.gamma = 0.99
        self.memory_size = 50000
        self.batch_size = 64
        self.exploration_rate = 1.0
        self.exploration_decay = 0.995
        self.tau = 0.005
        self.save_model = True
        self.run_id = np.random.randint(0, 1000)
        self.render = False

        torch.autograd.set_detect_anomaly(True)

    def loop(self, num_episodes=1000):
        reward_list = []
        ep_reward_mean = deque(maxlen=100)
        replay_buffer = deque(maxlen=self.memory_size)

        for episode in range(num_episodes):

            if episode % 10 == 0:
                seed = np.random.randint(0, 255)

            state = self.env.reset()
            ep_reward = 0

            while True:
                if self.render: self.env.render()
                action = self.get_action(state)
                new_state, reward, done, info = self.env.step(action)

                ep_reward += reward

                replay_buffer.append([state, action, reward, new_state, done])
                if done: break
                state = new_state

                self.update_networks(replay_buffer)
                self._update_target(self.critic.parameters(), self.critic_target.parameters(), tau=self.tau)

            self.exploration_rate = self.exploration_rate * self.exploration_decay if self.exploration_rate > 0.05 else 0.05
            ep_reward_mean.append(ep_reward)
            reward_list.append(ep_reward)
            if self.verbose > 0: print(
                f"Episode: {episode:7.0f}, reward: {ep_reward:8.2f}, mean_last_100: {np.mean(ep_reward_mean):8.2f}, exploration: {self.exploration_rate:0.2f}, goal: {info['goal_reached']}, collision: {info['collision']}")
            if self.verbose > 1: np.savetxt(f"data/reward_DDPG_PT_{self.run_id}.txt", reward_list)

    def get_action(self, state):

        state = state.reshape(1, -1)
        action = self.actor(T.tensor(state))
        action = action.detach().numpy()[0]

        action += np.random.normal(loc=0, scale=self.exploration_rate)

        return action

    def _update_target(self, weights, target_weights, tau):
        for (a, b) in zip(target_weights, weights):
            a.data.copy_(b * tau + a * (1 - tau))

    def update_networks(self, replay_buffer):
        samples = np.array(random.sample(replay_buffer, min(len(replay_buffer), self.batch_size)), dtype=object)



        # with tf.GradientTape() as tape_a, tf.GradientTape() as tape_c:

        objective_function_a = self.actor_objective_function(samples)  # Compute loss with custom loss function
        self.actor_optimizer.zero_grad()
        objective_function_a.backward()
        self.actor_optimizer.step()

        objective_function_c = self.critic_objective_function(samples)  # Compute loss with custom loss function
        self.critic_optimizer.zero_grad()
        objective_function_c.backward()
        self.critic_optimizer.step()

        # grads_c = tape_c.gradient(objective_function_c,
        #                           self.critic.trainable_variables)  # Compute gradients critic for network
        # grads_a = tape_a.gradient(objective_function_a,
        #                           self.actor.trainable_variables)  # Compute gradients actor for network
        #
        # self.critic_optimizer.apply_gradients(
        #     zip(grads_c, self.critic.trainable_variables))  # Apply gradients to update network weights
        # self.actor_optimizer.apply_gradients(
        #     zip(grads_a, self.actor.trainable_variables))  # Apply gradients to update network weights

    def actor_objective_function(self, replay_buffer):
        # Extract values from buffer
        state = T.from_numpy(np.vstack(replay_buffer[:, 0])).float().to(self.device)
        action = T.from_numpy(np.vstack(replay_buffer[:, 1])).float().to(self.device)

        action = self.actor(state)
        target = self.critic(state, action)
        mne = T.mean(target)
        return - mne

    def critic_objective_function(self, replay_buffer, gamma=0.99):
        # Extract values from buffer
        state = T.from_numpy(np.vstack(replay_buffer[:, 0])).float().to(self.device)
        action = T.from_numpy(np.vstack(replay_buffer[:, 1])).float().to(self.device)
        reward = T.from_numpy(np.vstack(replay_buffer[:, 2])).float().to(self.device)
        new_state = T.from_numpy(np.vstack(replay_buffer[:, 3])).float().to(self.device)
        done = T.from_numpy(np.vstack(replay_buffer[:, 4]).astype(np.int8)).float().to(
            self.device)

        # Compute the objective function
        # => r + gamma * max(Q(s', a'))
        # L'idea sarebbe quella di utilizzare la Bellman, in particolare vorrei avere il q value dell'azione col valore più alto a pArteire dallo stato s'
        # Trovandomi però con un action sapce continuo non posso testare le diverse azioni
        # Prendo quindi come azione migliore quella che la policy avrebbe scelto in quello stato ,e suppongo sarebbe la scelta migliore
        best_action = self.actor(new_state)

        max_q = self.critic_target(new_state, best_action)

        target = reward + gamma * max_q * (1 - done)

        predicted_values = self.critic(state, action)
        mse = T.square(predicted_values - target)

        return T.mean(mse)

