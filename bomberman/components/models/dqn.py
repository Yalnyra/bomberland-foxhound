import collections
import random
import torch
import torch.nn as nn
import torch.nn.functional as F

DQN_AGENT_PATH = "agent_dqn.pt"

Transition = collections.namedtuple(
    'Transition',
    ('state', 'action', 'next_state', 'reward')
)

class ReplayMemory(object):
    def __init__(self, capacity: int):
        self.memory = collections.deque([], maxlen=capacity)

    def push(self, *args):
        self.memory.append(Transition(*args))

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)


class DQNAgent(nn.Module):
    def __init__(self, n_observations, n_actions):
        super(DQNAgent, self).__init__()
        self.layer1 = nn.Linear(n_observations, 64)
        self.layer2 = nn.Linear(64, 64)
        self.layer3 = nn.Linear(64, n_actions)

    def forward(self, state):
        x = F.relu(self.layer1(state))
        x = F.relu(self.layer2(x))
        x = self.layer3(x)
        return x

    def save(self):
        torch.save(self, DQN_AGENT_PATH)
   
    def show(self):
        print(self)
