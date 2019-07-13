import gym
import torch

from utils import plotGraph
from model import DQN
from agent import DQNAgent
from agent import DQfDAgent


if __name__ == "__main__":
    env = gym.make("CartPole-v0")
    # env = gym.make("RoadRunner-v0")
    obs = env.reset()
    in_size = obs.shape[0]
    out_size = env.action_space.n
    steps = 6000

    model_dqn = DQN(in_size, out_size, save_model_name="dqn")
    agent_dqn = DQNAgent(model_dqn, env, double_DQN=True,
                         replay_start=200, target_update_freq=0.005,
                         eps_start=0.9, eps_decay=1000)
    agent_dqn.train_test(num_step=steps, test_period=200, test_step=500)

    # model_demonstrator = DQN(in_size, out_size, save_model_name="")
    agent_demonstrator = DQNAgent(model_dqn, env, replay_start=50, target_update_freq=0.005)
    agent_demonstrator.memory.reset()
    agent_demonstrator.run(update_memory=True, num_step=5e3)

    model_dqfd = DQN(in_size, out_size, save_model_name="dqfd")
    agent_dqfd = DQfDAgent(model_dqfd, env, agent_demonstrator.memory, double_DQN=True,
                           replay_start=200, target_update_freq=0.005,
                           eps_start=0.9, eps_decay=1000, n_step=1,
                           demo_percent=0.3, lambda_1=0, lambda_2=0.05,
                           expert_margin=0.5)

    agent_dqfd.pre_train(2000)
    agent_dqfd.train_test(num_step=steps, test_period=200, test_step=500)

    plotGraph("dqn", "dqfd", "DQN", "DQfD")
