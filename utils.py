import pickle
import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict
from functools import reduce
import torch


class Transition(object):
    def __init__(self, state, action, reward, next_state):
        self.state = state
        self.action = action
        self.reward = reward
        self.next_state = next_state

    def extend(self, trans):
        self.state = np.concatenate([self.state, trans.state])
        self.action = np.concatenate([self.action, trans.action])
        self.reward = np.concatenate([self.reward, trans.reward])
        self.next_state = np.concatenate([self.next_state, trans.next_state])
        return self


class Container(object):
    def __init__(self, save_name):
        self.data = defaultdict(list)
        self.record = defaultdict(lambda: defaultdict(list))
        if save_name:
            self.save_name = save_name
        else:
            self.save_name = "temp"

    def add(self, key, val, timeStamp=None):
        self.data[key].append(val)
        if timeStamp is not None:
            self.record[key][timeStamp].append(val)

    def get(self, key):
        if key not in self.data:
            return 0

        result = np.mean(self.data[key])
        self.reset(key)
        return result

    def reset(self, key):
        self.data.pop(key, None)

    def save(self):
        with open("{}.pickle".format(self.save_name), "wb") as f:
            pickle.dump(dict(self.record), f)

    def load(self):
        with open("{}.pickle".format(self.save_name), "rb") as f:
            self.record = pickle.load(f)


class RewardStepPairs(object):
    def __init__(self, rewards=None, steps=None):
        if rewards is None and steps is None:
            self.rewards = []
            self.steps = []
        elif rewards is not None and steps is not None:
            self.rewards = rewards
            self.steps = steps
        else:
            raise Exception("")

    def push(self, reward, step):
        self.rewards.append(reward)
        self.steps.append(step)

    def reset(self):
        self.rewards = []
        self.steps = []


class ReplayMemory(object):
    def __init__(self, capacity, n_step=None, gamma=None):
        capacity = int(capacity)
        self.capacity = capacity
        self.states = np.array([None]*capacity)
        self.actions = np.array([None]*capacity)
        self.rewards = np.array([None]*capacity)
        self.position = 0
        self.size = 0
        self.n_step = n_step
        self.gamma = gamma
        if n_step is not None and gamma is None:
            raise Exception("")

    def push(self, state, action, reward):
        self.states[self.position] = state
        self.actions[self.position] = action
        self.rewards[self.position] = reward
        self.position = (self.position + 1) % self.capacity
        self.size += 1
        if self.size > self.capacity:
            self.size = self.capacity

    def sample(self, batch_size, n_step=None):
        if n_step is not None:
            self.n_step = n_step

        index = np.array([], dtype=int)
        while index.size < batch_size:
            index_ = np.random.choice(self.size, batch_size).astype(int)
            index_ = self._valid_index(index_)
            index = np.concatenate([index, index_])
        index = index[:batch_size]
        states_batch = self.states[index]
        actions_batch = self.actions[index]
        rewards_batch = self.rewards[index]
        next_states_batch = self.states[(index+1)%self.capacity]
        trans = Transition(states_batch, actions_batch, rewards_batch,
                   next_states_batch)
        if self.n_step is None:
            return trans
        else:
            n_returns = self.n_step_return(index)
            n_step_states = self.states[(index+self.n_step)%self.capacity]
            return trans, n_returns, n_step_states

    def _valid_index(self, index):
        if self.n_step is None:
            valid_index = [self.states[i] is not None for i in index]
            return index[valid_index]

        n = self.n_step

        def util(i):
            if i+n <= self.capacity:
                states = self.states[i:i+n]
            else:
                states = np.concatenate([self.states[i:],
                                         self.states[:i+n-self.capacity]])
            return np.all([state is not None for state in states])

        valid_index = [util(i) for i in index]
        return index[valid_index]

    def n_step_return(self, index):
        n = self.n_step

        def util(i):
            if i+n <= self.capacity:
                rewards = self.rewards[i:i+n]
            else:
                rewards = np.concatenate([self.rewards[i:],
                                          self.rewards[:i+n-self.capacity]])
            return reduce(lambda x,y: self.gamma*x+y, reversed(rewards))

        return [util(i) for i in index]

    def __len__(self):
        return self.size

    def reset(self):
        self.states = np.array([None] * self.capacity)
        self.actions = np.array([None] * self.capacity)
        self.rewards = np.array([None] * self.capacity)
        self.position = 0
        self.size = 0


def norm_col_init(weights, std=1.0):
    x = torch.randn(weights.size())
    x *= std / torch.sqrt((x**2).sum(1, keepdim=True))
    return x


def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        weight_shape = list(m.weight.data.size())
        fan_in = np.prod(weight_shape[1:4])
        fan_out = np.prod(weight_shape[2:4]) * weight_shape[0]
        w_bound = np.sqrt(6. / (fan_in + fan_out))
        m.weight.data.uniform_(-w_bound, w_bound)
        m.bias.data.fill_(0)
    elif classname.find('Linear') != -1:
        weight_shape = list(m.weight.data.size())
        fan_in = weight_shape[1]
        fan_out = weight_shape[0]
        w_bound = np.sqrt(6. / (fan_in + fan_out))
        m.weight.data.uniform_(-w_bound, w_bound)
        m.bias.data.fill_(0)


def plotGraph(file1, file2, name1, name2, filename):
    with open("{}.pickle".format(file1), "rb") as f:
        data1 = pickle.load(f)["reward"]

    with open("{}.pickle".format(file2), "rb") as f:
        data2 = pickle.load(f)["reward"]

    x1, y1 = [], []
    for k, v in iter(data1.items()):
        x1.append(k)
        y1.append(np.mean(v))

    x2, y2 = [], []
    for k, v in iter(data2.items()):
        x2.append(k)
        y2.append(np.mean(v))

    # to get mean value in txt
    mean_file = open("cache.txt", "a")
    mean_file.writelines(filename + " : \n")
    mean_file.writelines(file1 + " " + str(np.mean(y1)) + "\n")
    mean_file.writelines(file2 + " " + str(np.mean(y2)) + "\n")
    mean_file.writelines("\n")
    mean_file.close()

    plt.plot(x1, y1)
    plt.plot(x2, y2)
    plt.xlabel("Step")
    plt.ylabel("Training Episode Returns")
    plt.grid()
    plt.legend([name1, name2])
    plt.savefig(filename)
    plt.close()
    # plt.show()
