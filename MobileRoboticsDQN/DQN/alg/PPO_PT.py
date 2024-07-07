import torch
import torch as T
import torch.nn as nn
from collections import deque
import numpy as np
import torch.nn.functional as F
import gym
import random


class Network_cont(nn.Module):
    def __init__(self, input_shape, output_size, output_range, hiddenNodes=32):
        # Input -> 64 -> 64 -> output
        bound = 0.003
        super(Network_cont, self).__init__()
        self.input_layer = nn.Linear(in_features=input_shape, out_features=hiddenNodes)
        self.hidden = nn.Linear(in_features=hiddenNodes, out_features=hiddenNodes)
        self.hidden2 = nn.Linear(in_features=hiddenNodes, out_features=hiddenNodes)
        self.output_layer = nn.Linear(in_features=hiddenNodes,
                                      out_features=output_size)  # np.array(output_size).prod())
        # nn.init.uniform_(self.output_layer.weight, -bound, bound)

        self.output_range = output_range

    def forward(self, x):
        # x = nn.functional.relu(self.input_layer(x))
        x = self.input_layer(x)
        x = nn.functional.relu(self.hidden(x))
        x = nn.functional.relu(self.hidden2(x))

        res = nn.functional.sigmoid(self.output_layer(x))
        output = (res * T.tensor(self.output_range[1] - self.output_range[0]) + T.tensor(
            self.output_range[0]))  # in rangxe [0,1]

        return output


class Network_disc(nn.Module):
    def __init__(self, input_shape, output_size, hiddenNodes=64):
        # Input -> 64 -> 64 -> output
        super(Network_disc, self).__init__()
        self.input_layer = nn.Linear(in_features=input_shape, out_features=hiddenNodes)
        self.hidden = nn.Linear(in_features=hiddenNodes, out_features=hiddenNodes)
        self.hidden2 = nn.Linear(in_features=hiddenNodes, out_features=hiddenNodes)
        self.output_layer = nn.Linear(in_features=hiddenNodes,
                                      out_features=output_size)  # np.array(output_size).prod())

    def forward(self, x):
        # x = nn.functional.relu(self.input_layer(x))
        x = self.input_layer(x)
        x = nn.functional.relu(self.hidden(x))
        x = nn.functional.relu(self.hidden2(x))

        return nn.functional.softmax(self.output_layer(x))


class Network_critic(nn.Module):
    def __init__(self, input_shape, hiddenNodes=64):
        # Input -> 64 -> 64 -> output

        super(Network_critic, self).__init__()
        self.input_layer = nn.Linear(in_features=input_shape, out_features=hiddenNodes)
        self.hidden = nn.Linear(in_features=hiddenNodes, out_features=hiddenNodes)
        self.hidden2 = nn.Linear(in_features=hiddenNodes, out_features=hiddenNodes)
        self.output_layer = nn.Linear(in_features=hiddenNodes,
                                      out_features=1)  # np.array(output_size).prod())

    def forward(self, x):
        x = self.input_layer(x)
        x = nn.functional.relu(self.hidden(x))
        x = nn.functional.relu(self.hidden2(x))
        return self.output_layer(x)


class PPO_PT:
    def __init__(self, env, discrete, verbose):
        self.env = env
        self.discrete = discrete
        self.seed = np.random.randint(0, 1000)
        torch.manual_seed(self.seed)
        np.random.seed(self.seed)
        self.verbose = verbose
        self.device = T.device("cuda:0" if T.cuda.is_available() else "cpu")

        self.input_shape = self.env.observation_space.shape[0]
        if self.discrete:
            self.action_space = env.action_space.n
        else:
            self.action_space = env.action_space.shape[0]

        if self.discrete:
            self.actor = Network_disc(self.input_shape, self.action_space).to(self.device)
            self.get_action = self.get_action_disc
            self.actor_objective_function = self.actor_objective_function_disc
        else:
            self.actor = Network_cont(self.input_shape, self.action_space,
                                      [env.action_space.low, env.action_space.high]).to(self.device)
            self.get_action = self.get_action_cont
            self.actor_objective_function = self.actor_objective_function_cont

        self.critic = Network_critic(self.input_shape).to(self.device)

        self.actor_optimizer = T.optim.Adam(self.actor.parameters())
        self.critic_optimizer = T.optim.Adam(self.critic.parameters())
        self.gamma = 0.99
        self.sigma = 1
        self.exploration_decay = 0.999999
        self.batch_size = 128
        self.epoch = 40

        self.run_id = np.random.randint(0, 1000)
        self.render = False

    def loop(self, num_episodes=1000):
        reward_list = []
        success_list = []
        collision_list = []
        ep_reward_mean = deque(maxlen=100)
        memory_buffer = deque()

        for episode in range(num_episodes):

            state = self.env.reset()
            ep_reward = 0

            while True:
                if self.render: self.env.render()
                action, action_prob = self.get_action(state)
                new_state, reward, done, info = self.env.step(action)
                ep_reward += reward

                memory_buffer.append([state, action, action_prob, reward, new_state, done])
                if done and episode % 10 == 0:
                    self.update_networks(np.array(memory_buffer, dtype=object), self.epoch, self.batch_size)
                    memory_buffer.clear()

                if done: break
                state = new_state

            ep_reward_mean.append(ep_reward)
            success_list.append(int(info['goal_reached']))
            collision_list.append(int(info['collision']))
            reward_list.append(ep_reward)
            if self.verbose > 0 and not self.discrete: print(
                f"EpisodeR: {episode:7.0f}, reward: {ep_reward:8.2f}, mean_last_100: {np.mean(ep_reward_mean):8.2f}, sigma: {self.sigma:0.2f}")
            if self.verbose > 0 and self.discrete:
                print(f"EpisodeR: {episode:7.0f}, reward: {ep_reward:8.2f}, mean_last_100: {np.mean(ep_reward_mean):8.2f}, sigma: {self.sigma:0.2f}, goal: {info['goal_reached']}, collision: {info['collision']}")
                np.savetxt(f"MobileRoboticsDQN/DQN/model_testing/PPO_PT{self.seed}_reward.txt", reward_list)
                np.savetxt(f"MobileRoboticsDQN/DQN/model_testing/PPO_PT{self.seed}_collision.txt", collision_list)
                np.savetxt(f"MobileRoboticsDQN/DQN/model_testing/PPO_PT{self.seed}_success.txt", success_list)
            if self.verbose > 1 and episode % 1000 == 0:
                model_script = T.jit.script(self.actor)
                model_script.save("MobileRoboticsDQN/DQN/model_testing/PPO_MODEL_%d.pt" % (episode))

    def get_action_cont(self, state):
        state = state.reshape((1, -1))
        mu = self.actor(T.tensor(state)).detach().numpy()
        #print("MU ", mu)
        action = np.random.normal(loc=mu, scale=self.sigma)
        self.sigma *= self.exploration_decay  # PPO DAVIDE CORSI
        return action[0], mu[0]

    def actor_objective_function_cont(self, memory_buffer):
        # Extract values from buffer
        state = T.from_numpy(np.vstack(memory_buffer[:, 0])).float().to(self.device)
        action = T.from_numpy(np.vstack(memory_buffer[:, 1])).float().to(self.device)
        mu = T.from_numpy(np.vstack(memory_buffer[:, 2])).float().to(self.device)
        reward = T.from_numpy(np.vstack(memory_buffer[:, 3])).float().to(self.device)
        new_state = T.from_numpy(np.vstack(memory_buffer[:, 4])).float().to(self.device)
        done = T.from_numpy(np.vstack(memory_buffer[:, 5]).astype(np.int8)).to(self.device)

        baseline = self.critic(state)
        adv = self._Gt(reward, new_state, done) - baseline  # Advantage = TD - baseline
        predictions_mu = self.actor(state)

        # prob = T.sqrt(T.tensor(1 / (2 * np.pi * self.sigma ** 2))) * T.exp(T.tensor(-(action - predictions_mu) ** 2
        # / (2 * self.sigma ** 2)))

        inputp1 = 1 / (2 * np.pi * self.sigma ** 2)

        inputp1 = T.tensor(inputp1)

        prob = T.sqrt(inputp1) * T.exp(
            -(action - predictions_mu) ** 2 / (2 * self.sigma ** 2))

        # old_prob = T.sqrt(T.tensor(1 / (2 * np.pi * self.sigma ** 2))) * np.math.e ** (-(action - mu) ** 2 / (2 *
        # self.sigma ** 2))

        old_prob = T.sqrt(inputp1) * np.math.e ** (
                -(action - mu) ** 2 / (2 * self.sigma ** 2))

        prob = T.mean(prob, dim=1, keepdim=True)
        old_prob = T.mean(old_prob, dim=1, keepdim=True)

        r_theta = T.div(prob, old_prob)

        clip_val = 0.2
        obj_1 = r_theta * adv
        obj_2 = T.clamp(r_theta, 1 - clip_val, 1 + clip_val) * adv

        # partial_objective = tf.math.minimum(obj_1, obj_2)
        # partial_objective = T.min(obj_1, obj_2)
        partial_objective = T.minimum(obj_1, obj_2)
        # mean = T.mean(partial_objective)
        # return -mean
        return -T.mean(partial_objective)

    def _Gt(self, reward, new_state, done):
        # print(type(T.tensor(reward)), type(1-done))
        return T.tensor(reward) + (1 - done) * self.gamma * self.critic(
            new_state)  # 1-Step TD, for the n-Step TD we must save more sequence in the buffer

    ### CRITIC METHODS ###
    def critic_objective_function(self, memory_buffer):
        # Extract values from buffer
        state = T.from_numpy(np.vstack(memory_buffer[:, 0])).float().to(self.device)
        reward = np.vstack(memory_buffer[:, 3])
        new_state = T.from_numpy(np.vstack(memory_buffer[:, 4])).float().to(self.device)
        done = T.from_numpy(np.vstack(memory_buffer[:, 5]).astype(np.int8)).to(self.device)

        predicted_value = self.critic(state)
        target = self._Gt(reward, new_state, done)
        mse = T.square(predicted_value - target)

        return T.mean(mse)

    def update_networks(self, memory_buffer, epoch, batch_size):
        # batch_size = min(len(memory_buffer), batch_size)
        # mini_batch_n = int(len(memory_buffer) / batch_size)
        # batch_list = np.array_split(memory_buffer, mini_batch_n)

        for _ in range(epoch):
            idx = np.random.randint(memory_buffer.shape[0], size=self.batch_size)
            training_batch = memory_buffer[idx]

            objective_function_c = self.critic_objective_function(
                training_batch)  # Compute loss with custom loss function
            self.critic_optimizer.zero_grad()
            objective_function_c.backward()
            self.critic_optimizer.step()

            objective_function_a = self.actor_objective_function(
                memory_buffer)  # Compute loss with custom loss function
            self.actor_optimizer.zero_grad()
            objective_function_a.backward()
            self.actor_optimizer.step()

    # DISCRETE

    def get_action_disc(self, state):
        state = state.reshape(1, -1)
        softmax_out = self.actor(T.tensor(state))

        selected_action = np.random.choice(self.action_space, p=softmax_out.detach().numpy()[0])

        return selected_action, softmax_out.detach().numpy()[0][selected_action]

    def actor_objective_function_disc(self, memory_buffer):
        # Extract values from buffer
        state = T.from_numpy(np.vstack(memory_buffer[:, 0])).float().to(self.device)
        action = T.tensor(np.vstack(memory_buffer[:, 1])).to(self.device)
        action_prob = T.tensor(np.vstack(memory_buffer[:, 2])).to(self.device)
        reward = np.vstack(memory_buffer[:, 3])
        new_state = T.from_numpy(np.vstack(memory_buffer[:, 4])).float().to(self.device)
        done = T.from_numpy(np.vstack(memory_buffer[:, 5])).float().to(self.device)

        baseline = self.critic(state)
        adv = self._Gt(reward, new_state, done) - baseline  # Advantage = TD - baseline

        prob = self.actor(state)

        action_idx = []
        for val in action:
            action_idx.append([val])

        action_idx = T.tensor(action_idx)
        prob = T.gather(prob, dim=1, index=action_idx)
        # prob = T.unsqueeze(prob, dim=-1)

        r_theta = T.div(prob, action_prob)
        clip_val = 0.2
        obj_1 = r_theta * adv

        obj_2 = T.clamp(r_theta, 1 - clip_val, 1 + clip_val) * adv

        partial_objective = T.min(obj_1, obj_2)

        mean = T.mean(partial_objective)
        return -mean
