import torch
import torch.nn as nn
import torch.optim as optim
import random
from collections import deque
import pickle
import pandas as pd
import numpy as np

print(torch.__version__)

class DQLAgent(nn.Module):
    def __init__(self, action_size, state_size, qweights_1, qweights_2, multi_weight, batch, learning_rate, round_exp,
                 gamma=0.6, epsilon=0.1, neural_model=0, id=None, path="", freeze_n_layers=False):
        super(DQLAgent, self).__init__()

        torch.cuda.empty_cache()

        print("\n\n--------------------------\nLEARNING RATE: -> ", +learning_rate, "\n---------------------------\n")

        self._state_size = state_size
        self._action_size = action_size
        self.gamma = gamma
        self.epsilon = epsilon
        self.neural_model = neural_model
        self.multi_weight = float(multi_weight)
        self.batch = batch
        self.experience_replay = deque(maxlen=1000000)

        self.freeze_n_layers = freeze_n_layers
                
        self.path = path
        self.id = id
        self.round_exp = round_exp

        self.q_network_1 = self._build_compile_model()
        self.target_network_1 = self._build_compile_model()
        self.q_network_2 = self._build_compile_model()
        self.target_network_2 = self._build_compile_model()

        self.optimizer_1 = optim.Adam(self.q_network_1.parameters(), lr=learning_rate)
        self.optimizer_2 = optim.Adam(self.q_network_2.parameters(), lr=learning_rate)        

        if qweights_1:
            print("\n*** DQL LOADING PREVIOUS NETWORKS 2 WEIGHTS")
            self.q_network_1.load_state_dict(qweights_1)
            self.target_network_1.load_state_dict(qweights_1)

            self.q_network_2.load_state_dict(qweights_2)
            self.target_network_2.load_state_dict(qweights_2)

        self.mean_loss_list_1 = []
        self.mse_list_1 = []
        self.mean_loss_list_2 = []
        self.mse_list_2 = []
        self.gradient_steps = []

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.to(self.device)

    def _build_compile_model(self):
        if self.neural_model == 0:  # standard
            model = nn.Sequential(
                nn.Linear(self._state_size, 32),
                nn.ReLU(),
                nn.Linear(32, 32),
                nn.ReLU(),
                nn.Linear(32, self._action_size)
            )
            model.to(self.device)

        if self.freeze_n_layers:
            for param in model[:-1].parameters():
                param.requires_grad = False
                
            print(" ((*)) FREEZING LAYERS ((*))", self.id)

        return model

    def align_target_model(self):
        self.target_network_1.load_state_dict(self.q_network_1.state_dict())
        self.target_network_2.load_state_dict(self.q_network_2.state_dict())

    def predict(self, rawstate):
        state = torch.tensor(rawstate, dtype=torch.float32).to(self.device)
        q_values_1 = self.q_network_1(state)
        q_values_2 = self.q_network_2(state)

        q_values_mult = self.multi_weight * q_values_1 + (1 - self.multi_weight) * q_values_2

        return torch.argmax(q_values_mult).item()

    def store(self, state, action, reward, next_state, terminated, info):
        state = torch.tensor(state, dtype=torch.float32).to(self.device)
        next_state = torch.tensor(next_state, dtype=torch.float32).to(self.device)
        action = torch.tensor(action, dtype=torch.long).to(self.device)
        reward = torch.tensor(reward, dtype=torch.float32).to(self.device)
        terminated = torch.tensor(terminated, dtype=torch.float32).to(self.device)
        self.experience_replay.append((state, action, reward, next_state, terminated, info))

    def retrain(self, exp_replay=None, apply_old_replay_buffer=False):
        print("\n\n################\nRETRAINING\n###############")

        if not exp_replay:
            minibatch = random.sample(self.experience_replay, min(self.batch, len(self.experience_replay)))
            print("length minibatch:", len(minibatch), len(self.experience_replay))
        else:
            minibatch = random.sample(exp_replay, min(self.batch, len(exp_replay)))
            print("length custom minibatch:", len(minibatch), len(exp_replay))

        if(apply_old_replay_buffer):
            print("applying old replay buffer")
            minibatch = minibatch + random.sample(apply_old_replay_buffer, len(minibatch))
            print("length minibatch combined with old:", len(minibatch), len(self.experience_replay))

        mean_loss_1 = 0
        mse_1 = 0
        mean_loss_2 = 0
        mse_2 = 0
        steps = 0

        print("Training...")

        for state, action, reward, next_state, terminated, _ in minibatch:
            try:
                target_1 = self.q_network_1(state)
                target_2 = self.q_network_2(state)

                if terminated:
                    print("terminated round in retrain")
                    target_1[0][action] = reward[0]
                    target_2[0][action] = reward[1]
                else:
                    t_1 = self.target_network_1(next_state).max(1)[0].unsqueeze(1)
                    t_2 = self.target_network_2(next_state).max(1)[0].unsqueeze(1)

                    target_1[0][action] = reward[0] + self.gamma * t_1
                    target_2[0][action] = reward[1] + self.gamma * t_2

                loss_1 = nn.functional.smooth_l1_loss(self.q_network_1(state), target_1)
                loss_2 = nn.functional.smooth_l1_loss(self.q_network_2(state), target_2)

                self.optimizer_1.zero_grad()
                self.optimizer_2.zero_grad()
                loss_1.backward()
                loss_2.backward()
                self.optimizer_1.step()
                self.optimizer_2.step()

                steps += 1
                mean_loss_1 += loss_1.item()
                mse_1 += nn.functional.mse_loss(self.q_network_1(state), target_1).item()

                mean_loss_2 += loss_2.item()
                mse_2 += nn.functional.mse_loss(self.q_network_2(state), target_2).item()

            except Exception as e:
                print("ERROR", e)

        try:
            self.mean_loss_list_1.append(mean_loss_1 / steps)
            self.mse_list_1.append(mse_1 / steps)

            self.mean_loss_list_2.append(mean_loss_2 / steps)
            self.mse_list_2.append(mse_2 / steps)
            self.gradient_steps.append(steps)
        except:
            pass

        self.align_target_model()

    def saveBuffer(self):
        pickle.dump(self.experience_replay, open(self.path + str(self.id) + str(self.round_exp) + "_replaybuffer.npy", 'wb'))
        pd.DataFrame([[self.mean_loss_list_1, self.mse_list_1, self.mean_loss_list_2, self.mse_list_2,
                       self.gradient_steps]]).to_csv(self.path + str(self.id) + str(self.round_exp) + "_QR_TR_LOSS.bz2",
                                                    index=False, mode='a', header=False, compression="bz2")

    def loadBuffer(self, rbuff=False):
        if(rbuff):
            self.experience_replay = rbuff
        else:
            self.experience_replay = pickle.load(open(self.path + str(self.id) + str(self.round_exp) + "_replaybuffer.npy", 'rb'))