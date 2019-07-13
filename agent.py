import torch.nn as nn
import numpy as np
import copy
import torch
from torch.autograd import Variable

from utils import ReplayMemory
from utils import RewardStepPairs
from utils import Container

LongTensor = torch.LongTensor
FloatTensor = torch.FloatTensor
ByteTensor = torch.ByteTensor
Tensor = FloatTensor


class Agent(object):
    def __init__(self, **kwargs):
        self.BATCH_SIZE = kwargs.pop("batch_size", 32)
        self.GAMMA = kwargs.pop("gamma", 0.99)
        self.EPS_START = kwargs.pop("eps_start", 0.95)
        self.EPS_END = kwargs.pop("eps_end", 0.01)
        self.EPS_DECAY = kwargs.pop("eps_decay", 2000)
        self.REPLAY_CAPACITY = kwargs.pop("replay_capacity", 50000)
        self.REPLAY_START = kwargs.pop("replay_start", 2000)
        self.DOUBLE_DQN = kwargs.pop("double_DQN", False)
        self.TARGET_UPDATE_FREQ = kwargs.pop("target_update_freq", 0.01)
        self.ACTION_REPEAT = kwargs.pop("action_repeat", 1)
        self.LR = kwargs.pop("learning_rate", 1e-3)
        self.GRADIENT_CLIPPING = kwargs.pop("gradient_clipping", 1)
        self.VERBOSE = kwargs.pop("verbose", False)
        self.GET_DEMO = kwargs.pop("get_demo", False)
        self.rule_processor = kwargs.pop("rule_processor", None)
        self.state_processor = kwargs.pop("state_processor", None)
        if self.TARGET_UPDATE_FREQ < 1:
            self.SOFT_UPDATE = True
            self.TARGET_UPDATE_FREQ = float(self.TARGET_UPDATE_FREQ)
        else:
            self.SOFT_UPDATE = False
            self.TARGET_UPDATE_FREQ = int(self.TARGET_UPDATE_FREQ)

        self.i_step = 0
        self.i_episode = 0
        self.record_i_step = 0
        self.record_i_episode = 0
        self.is_render = False
        self.is_training = True
        self.is_test = False
        self.reward_step_pairs = RewardStepPairs()


class DQNAgent(Agent):
    def __init__(self, model, env, **kwargs):
        Agent.__init__(self, **kwargs)
        self.update_step = 0
        self.eps = self.EPS_START
        self.global_step = 0
        self.model = model
        self.target_model = copy.deepcopy(model)
        self.in_size = model.in_size
        self.out_size = model.out_size
        self.memory = ReplayMemory(self.REPLAY_CAPACITY)
        self.opt = torch.optim.Adam(self.model.parameters(), lr=self.LR)
        self.env = env
        self.container = Container(self.model.SAVE_MODEL_NAME)

    def select_action(self, state):
        if self.is_training:
            self.global_step += 1
            self.eps = self.EPS_START - (self.EPS_START-self.EPS_END)/self.EPS_DECAY * self.global_step
            if self.eps < self.EPS_END:
                self.eps = self.EPS_END

        if self.is_training and np.random.rand() < self.eps:
            return LongTensor([[np.random.randint(self.out_size)]])
        else:
            var = Variable(state).type(FloatTensor)
            out = self.model(var)
            return out.max(1)[1].data.view(1, 1)

    def _DQ_loss(self, y_pred, reward_batch, non_final_mask, non_final_next_states):
        q_next = Variable(torch.zeros(self.BATCH_SIZE).type(FloatTensor))
        target_q = self.target_model(non_final_next_states)
        if self.DOUBLE_DQN:
            max_act = self.model(non_final_next_states).max(1)[1].view(-1,1)
            q_next[non_final_mask] = target_q.gather(1, max_act).data.view(
                target_q.gather(1, max_act).data.shape[0]
            )
        else:
            q_next[non_final_mask] = target_q.max(1)[0].data

        # next_state_values.volatile = False
        y = q_next * self.GAMMA + reward_batch
        loss = nn.functional.mse_loss(y_pred, y)
        return loss

    def _calc_loss(self):
        batch = self.memory.sample(self.BATCH_SIZE)
        non_final_mask = ByteTensor(
            tuple([s is not None for s in batch.next_state]))
        non_final_next_states = Variable(torch.cat([s for s in batch.next_state if s is not None]))

        state_batch = Variable(torch.cat([s for s in batch.state if s is not None]))
        action_batch = Variable(torch.cat([s for s in batch.action if s is not None]))
        reward_batch = Variable(torch.cat([s for s in batch.reward if s is not None]))

        y_pred = self.model(state_batch).gather(1, action_batch).squeeze()
        loss = self._DQ_loss(y_pred, reward_batch, non_final_mask, non_final_next_states)
        self.container.add("y_pred", torch.mean(y_pred.data))
        self.container.add("loss", loss.data.item())
        return loss

    def update_policy(self):
        loss = self._calc_loss()
        self.opt.zero_grad()
        loss.backward()
        if self.GRADIENT_CLIPPING:
            for param in self.model.parameters():
                param.grad.data.clamp_(-self.GRADIENT_CLIPPING,
                                       self.GRADIENT_CLIPPING)
        self.opt.step()

    def update_target_network(self):
        if not self.SOFT_UPDATE:
            self.update_step = (self.update_step + 1) % self.TARGET_UPDATE_FREQ
            if self.update_step == 0:
                state_dict = self.model.state_dict()
                self.target_model.load_state_dict(copy.deepcopy(state_dict))
        else:
            tw = self.target_model.state_dict().values()
            sw = self.model.state_dict().values()
            for t, s in zip(tw, sw):
                t.add_(self.TARGET_UPDATE_FREQ*(s-t))

    def _forward(self, obs, is_train, update_memory):
        if self.state_processor:
            state = self.state_processor(obs)
        else:
            temp = obs[None, :] if len(obs.shape)==1 else obs[None, None, :]
            state = torch.from_numpy(temp).type(FloatTensor)

        if self.GET_DEMO:
            action = self.rule_processor(obs)
        else:
            action = self.select_action(state)

        act = action.numpy().squeeze()
        if self.VERBOSE:
            print("action: {}".format(act))
        action_step = self.ACTION_REPEAT
        reward = 0
        done = False
        while action_step > 0:
            action_step -= 1
            next_obs, r, done, _ = self.env.step(act)
            reward += r
            if done:
                break

        self.reward_episode += reward
        if update_memory:
            reward = FloatTensor([reward])
            self.memory.push(state, action, reward)
            if done:
                self.memory.push(None, None, None)

        if len(self.memory) >= self.REPLAY_START and is_train:
            self.update_policy()
            self.update_target_network()

        if self.is_render:
            self.env.render()

        return next_obs, done

    def fit(self, is_train, update_memory=True, num_step=np.inf, num_episode=np.inf, max_episode_length=np.inf, is_render=False):
        if num_step == np.inf and num_episode == np.inf:
            raise Exception("")
        if num_step != np.inf and num_episode != np.inf:
            raise Exception("")

        self.is_render = is_render
        while self.i_episode < num_episode and self.i_step < num_step:
            self.i_episode += 1
            print("------------------------")
            print("episode: {}, step: {}".format(self.i_episode, self.i_step))
            obs = self.env.reset()
            self.reward_episode = 0
            episode_step = 0
            while episode_step < max_episode_length:
                episode_step += 1
                self.i_step += 1
                obs, done = self._forward(obs, is_train, update_memory)
                if done:
                    self.reward_step_pairs.push(self.reward_episode, self.i_step)
                    if self.is_test:
                        self.container.add("reward", self.reward_episode, self.record_i_step)
                    self.print(is_train)
                    break

    def train(self, **kwargs):
        self.is_training = True
        if kwargs.pop("clear", True):
            self.i_episode = 0
            self.i_step = 0
            self.reward_step_pairs.reset()
        print("Training starts...")
        self.fit(True, **kwargs)
        # self.model.save()
        self.container.save()

    def run(self, **kwargs):
        self.is_training = False
        if kwargs.pop("clear", True):
            self.i_episode = 0
            self.i_step = 0
            self.reward_step_pairs.reset()
        print("Running starts...")
        self.fit(False, **kwargs)

    def _test(self, num_step):
        self.record_i_episode = self.i_episode
        self.record_i_step = self.i_step
        self.is_test = True
        self.run(num_step=num_step)
        self.i_episode = self.record_i_episode
        self.i_step = self.record_i_step
        self.is_test = False

    def train_test(self, num_step, test_period=1000, test_step=100):
        self.i_episode = 0
        self.i_step = 0
        while self.i_step < num_step:
            self._test(test_step)
            self.train(num_step=self.record_i_step+test_period, clear=False)
        self._test(test_step)

    def print(self, is_train):
        print("reward_episode {}".format(self.reward_episode))
        print("eps {}".format(self.eps))
        if is_train:
            print("loss_episode {}".format(self.container.get("loss")))
            print("y_pred_episode {}".format(self.container.get("y_pred")))


class DQfDAgent(DQNAgent):
    def __init__(self, model, env, demo_memory, **kwargs):
        DQNAgent.__init__(self, model, env, **kwargs)
        self.EXPERT_MARGIN = kwargs.pop("expert_margin", 0.8)
        self.DEMO_PER = kwargs.pop("demo_percent", 0.3)
        self.N_STEP = kwargs.pop("n_step", 5)
        self.LAMBDA_1 = kwargs.pop("lambda_1", 0.1)
        self.LAMBDA_2 = kwargs.pop("lambda_2", 0.5)
        self.LAMBDA_3 = kwargs.pop("lambda_3", 0)
        self.memory = ReplayMemory(self.REPLAY_CAPACITY, self.N_STEP, self.GAMMA)
        self.demo_memory = demo_memory
        self.demo_memory.n_step = self.N_STEP
        self.demo_memory.gamma = self.GAMMA
        self.is_pre_train = False

    def _n_step_loss(self, y_pred, n_returns_batch, non_final_n_mask, non_final_n_states):
        q_n = Variable(torch.zeros(self.BATCH_SIZE).type(FloatTensor))
        target_q_n = self.target_model(non_final_n_states)
        if self.DOUBLE_DQN:
            max_act_n = self.model(non_final_n_states).max(1)[1].view(-1, 1)
            q_n[non_final_n_mask] = target_q_n.gather(1, max_act_n).data.view(
                target_q_n.gather(1, max_act_n).data.shape[0]
            )
        else:
            q_n[non_final_n_mask] = target_q_n.max(1)[0].data

        y_n_step = q_n * np.power(self.GAMMA, self.N_STEP) + n_returns_batch
        return nn.functional.mse_loss(y_pred, y_n_step)

    def _expert_loss(self, q_pred, action_batch, non_demo_mask):
        y_pred = q_pred.gather(1, action_batch).squeeze()
        expert_margin = torch.zeros(self.BATCH_SIZE, self.out_size)
        expert_margin[:, action_batch.data] = self.EXPERT_MARGIN
        q_l = q_pred + Variable(expert_margin)
        j_e = q_l.max(1)[0] - y_pred
        j_e[non_demo_mask] = 0
        return j_e.sum()

    def _collect_batch(self):
        non_demo_mask = ByteTensor([False] * self.BATCH_SIZE)
        if self.is_pre_train:
            batch, n_returns, n_step_states = self.demo_memory.sample(
                self.BATCH_SIZE)
        else:
            demo_num = int(self.BATCH_SIZE * self.DEMO_PER)
            replay_demo, n_returns_demo, n_step_states_demo = \
                self.demo_memory.sample(demo_num)
            replay_agent, n_returns_agent, n_step_states_agent = \
                self.memory.sample(self.BATCH_SIZE - demo_num)
            batch = replay_demo.extend(replay_agent)
            if demo_num != self.BATCH_SIZE:
                non_demo_mask[demo_num:] = 1
            n_returns_demo.extend(n_returns_agent)
            n_returns = n_returns_demo
            n_step_states = np.concatenate([n_step_states_demo,
                                            n_step_states_agent])

        return batch, n_returns, n_step_states, non_demo_mask

    def _calc_loss(self):
        batch, n_returns, n_step_states, non_demo_mask = self._collect_batch()

        non_final_mask = ByteTensor(
            tuple([s is not None for s in batch.next_state]))
        non_final_next_states = Variable(torch.cat([s for s in batch.next_state if s is not None]))
        non_final_n_mask = ByteTensor(tuple([s is not None for s in
                                             n_step_states]))
        non_final_n_states = Variable(torch.cat([s for s in n_step_states if s is not None]))

        state_batch = Variable(torch.cat([s for s in batch.state if s is not None]))
        action_batch = Variable(torch.cat([s for s in batch.action if s is not None]))
        reward_batch = Variable(torch.cat([s for s in batch.reward if s is not None]))
        n_returns_batch = Variable(torch.cat(n_returns))

        q_pred = self.model(state_batch)
        y_pred = q_pred.gather(1, action_batch).squeeze()

        dq_loss = self._DQ_loss(y_pred, reward_batch, non_final_mask, non_final_next_states)
        n_step_loss = self._n_step_loss(y_pred, n_returns_batch,
                                        non_final_n_mask, non_final_n_states)
        expert_loss = self._expert_loss(q_pred, action_batch, non_demo_mask)
        loss = dq_loss + self.LAMBDA_1 * n_step_loss + self.LAMBDA_2 * expert_loss
        self.container.add("dq_loss", torch.mean(dq_loss.data))
        self.container.add("expert_loss", torch.mean(expert_loss.data))
        self.container.add("y_pred", torch.mean(y_pred.data))
        self.container.add("loss", torch.mean(loss.data))
        return loss

    def pre_train(self, steps):
        self.i_episode = 0
        self.i_step = 0
        self.is_pre_train = True
        print("Pre training...")
        for i in range(steps):
            if i % 500 == 0:
                print("Pre train steps: {}".format(i))
            self.update_policy()
            self.update_target_network()
        print("Pre train done")
        self.is_pre_train = False
