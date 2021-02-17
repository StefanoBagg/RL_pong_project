import numpy as np
import torch
import torch.nn.functional as F
from torch.distributions import Categorical
import os
import random
import copy
# from PIL import Image


#################################################
# IMPORTANT: CHANGE THE PATH OF THE MODEL BELOW #
#################################################
PATH_MODEL_TO_LOAD = "model.mdl"


class Policy(torch.nn.Module):
    def __init__(self, action_space=3, device=torch.device("cuda")):
        super(Policy, self).__init__()
        self.device = device # GPU preferred
        self.num_frames = 2
        self.action_space = action_space
        self.ep_clip = 0.1 # used for clamping the loss function
        self.hidden_neurons = 128 # network size parameter
        self.reshaped_size = 3520 # network size parameter

        # Network layers 2 Conv + 2 FC
        self.conv1 = torch.nn.Conv2d(self.num_frames, 16, 8, 4)
        self.conv2 = torch.nn.Conv2d(16, 32, 4, 2)
        self.fc1 = torch.nn.Linear(3520, self.hidden_neurons)
        self.fc2 = torch.nn.Linear(self.hidden_neurons, self.action_space)

        # Uncomment next line to load an existing model from PATH_MODEL_TO_LOAD during TRAINING
        self.init_weights(PATH_MODEL_TO_LOAD[1:])

    # Load weight from specified file PATH_MODEL_TO_LOAD
    def init_weights(self, model_path):
        try:
            weights = torch.load(model_path, map_location=self.device)
            self.load_state_dict(weights, strict=False)
            print("Model loaded from {}".format(PATH_MODEL_TO_LOAD))
        except FileNotFoundError:
            print("Model not found. Check the path and try again.")

    # Forward propagate the input in the network
    def layers(self, x):
        x = self.conv1(x)
        x = F.relu(x)
        x = self.conv2(x)
        # x = x.reshape(-1, self.reshaped_size)
        flat_size = x.size()[0]
        x = x.view(flat_size, -1)
        x = F.relu(x)
        x = self.fc1(x)
        x = F.relu(x)
        return self.fc2(x)

    # input: batches form Agent.episode_finished()
    # output: loss
    def forward(self, obs_batch, action_batch, action_prob_batch, advantage_batch):
        idm_action = np.array([[1., 0., 0.], [0., 1., 0.], [0., 0., 1.]])
        ts = torch.FloatTensor(idm_action[action_batch.cpu().numpy()]).to(self.device)

        logs = self.layers(obs_batch)
        r = torch.sum(F.softmax(logs, dim=1) * ts, dim=1) / action_prob_batch
        loss_1 = r * advantage_batch
        loss_2 = torch.clamp(r, 1 - self.ep_clip, 1 + self.ep_clip) * advantage_batch
        loss = -torch.min(loss_1, loss_2)
        loss = torch.mean(loss)

        return loss


class Agent(object):
    def __init__(self, player_id=1, evaluation=True, device_name="cpu"):
        self.name = "Agent_ppo_CNN"
        self.device = torch.device(device_name)
        self.policy = Policy(device=self.device).to(self.device)
        self.optimizer = torch.optim.Adam(self.policy.parameters(), lr=1e-4)  # First train 94k
        # self.optimizer = torch.optim.Adam(self.policy.parameters(), lr=1e-5)  # Second train 14k
        self.player_id = player_id
        self.gamma = 0.99

        self.evaluation = evaluation
        if self.evaluation:
            self.policy.eval()

        # Declare lists
        self.states = []
        self.actions = []
        self.action_probs = []
        self.rewards = []

        self.discounted_rewards = []

        self.pp_observation = None
        self.previous_observation = None

    def reset(self):
        self.previous_observation = None

    # Reset lists
    def reset_lists(self):
        self.states = []
        self.actions = []
        self.action_probs = []
        self.rewards = []

    def get_name(self):
        return self.name

    # Save model as "model_episode.mdl"
    def save_model(self, output_directory, episode=0):
        if not os.path.exists(output_directory):
            os.makedirs(output_directory)
        f_name = "{}/model_{}.mdl".format(output_directory, episode)
        torch.save(self.policy.state_dict(), f_name)

    # Load model to continue training, PATH_MODEL_TO_LOAD on top
    def load_model(self):
        self.policy.init_weights(PATH_MODEL_TO_LOAD)

    # Preprocess the image
    def preprocess(self, observation):
        observation = observation[:, 8:192]  # Crop pixels after ball passes paddle
        observation = observation[::2, ::2]  # Downsample the picture by factor of 2

        # Erase background in all the three channels
        observation[observation == 58] = 0
        observation[observation == 43] = 0
        observation[observation == 48] = 0

        # DEBUG plot image
        # img = Image.fromarray(observation, 'RGB')
        # # img.save('my.png')
        # img.show()

        # Squeeze 3 channels in 1 through mean
        observation = observation[::, ::].mean(axis=-1)

        # Assure that previous obs is not empty
        if self.previous_observation is None:
            self.previous_observation = copy.deepcopy(observation[np.newaxis, np.newaxis, :, :])

        # Stack current and past states
        observation = observation[np.newaxis, np.newaxis, :, :]
        stack_ob = np.concatenate((observation, self.previous_observation), axis=1)
        self.previous_observation = copy.deepcopy(observation)
        return torch.tensor(stack_ob, device=self.device, dtype=torch.float)

    # input:
    #       observation (state)
    # output:
    #       Evaluation True: action with maximum probability (for test)
    #       Evaluation False: sampled action and probability (for training)
    def get_action(self, observation):
        self.pp_observation = self.preprocess(observation)
        logits = self.policy.layers(self.pp_observation)
        if self.evaluation:
            action = int(torch.argmax(logits[0]).detach().cpu().numpy())
            return action
        else:
            c = torch.distributions.Categorical(logits=logits)
            action = int(c.sample().cpu().numpy()[0])
            action_prob = float(c.probs[0, action].detach().cpu().numpy())
            return action, action_prob

    # Store
    # - the state,
    # - the taken action,
    # - the taken action probability,
    # - reward
    def store_outcome(self, action, action_prob, reward):
        self.states.append(self.pp_observation)
        self.actions.append(action)
        self.action_probs.append(action_prob)
        self.rewards.append(reward)

    # Weights update
    # Step 1 - Compute discounted rewards
    # Step 2 - Update network 5 times
        # sample store_outcome(action, action_prob, reward) and advantage (computed as normalized discounted rewards)
        # use samples for compute loss with policy.forward()
        # use loss for backpropagation on network through optimizer
    def episode_finished(self):
        d_rew = 0
        self.discounted_rewards = []
        for rew in self.rewards[::-1]:
            if not rew == 0:
                d_rew = 0
            d_rew = rew + self.gamma * d_rew
            self.discounted_rewards.insert(0, d_rew)

        self.discounted_rewards = torch.FloatTensor(self.discounted_rewards)
        self.discounted_rewards = (self.discounted_rewards - self.discounted_rewards.mean()) / self.discounted_rewards.std()

        for _ in range(5):
            n_batch = 20000
            idxs = random.sample(range(len(self.actions)), min(len(self.actions), n_batch))
            obs_batch = torch.cat([self.states[idx] for idx in idxs], 0).to(self.device)
            action_batch = torch.LongTensor([self.actions[idx] for idx in idxs]).to(self.device)
            action_prob_batch = torch.FloatTensor([self.action_probs[idx] for idx in idxs]).to(self.device)
            advantage_batch = torch.FloatTensor([self.discounted_rewards[idx] for idx in idxs]).to(self.device)

            self.optimizer.zero_grad()
            loss = self.policy.forward(obs_batch, action_batch, action_prob_batch, advantage_batch)
            loss.backward()
            self.optimizer.step()
            del obs_batch, action_batch, action_prob_batch, advantage_batch

