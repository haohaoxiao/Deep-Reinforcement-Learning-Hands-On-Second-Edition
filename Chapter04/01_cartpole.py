#!/usr/bin/env python3
import gym
from collections import namedtuple
import numpy as np
from tensorboardX import SummaryWriter

import torch
import torch.nn as nn
import torch.optim as optim

# 隐藏层节点数
HIDDEN_SIZE = 128
# 多少个episode一个批次，目标是增加训练样本数
BATCH_SIZE = 16
# 设置reward排位多少以上的训练样本参与训练
PERCENTILE = 70

# 定义用于近似policy的NN
class Net(nn.Module):
    def __init__(self, obs_size, hidden_size, n_actions):
        super(Net, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, n_actions)
        )

    def forward(self, x):
        return self.net(x)

# 记录每个episode的reward和总steps
Episode = namedtuple('Episode', field_names=['reward', 'steps'])
# 记录episode中每一步的state和action
EpisodeStep = namedtuple('EpisodeStep', field_names=['observation', 'action'])

# 用当前的network跑一个批次的训练
def iterate_batches(env, net, batch_size):
    # 存储整个批次里的所有episode, while每循环一次，添加一个
    batch = []
    # 记录每个episode的reward
    episode_reward = 0.0
    # 记录每个episode所用的步数
    episode_steps = []
    # 重置env，获得state
    obs = env.reset()
    # 对action计算概率
    sm = nn.Softmax(dim=1)
    while True:
        # 将array转换成tensor
        obs_v = torch.FloatTensor([obs])
        # 计算各种action的概率
        act_probs_v = sm(net(obs_v))
        # 将tensor转换成array
        act_probs = act_probs_v.data.numpy()[0]
        # 根据概率随机抽取一个action
        action = np.random.choice(len(act_probs), p=act_probs)
        # 根据action获得下一步的state，本次的action以及game是否done
        next_obs, reward, is_done, _ = env.step(action)
        # 更新总reward
        episode_reward += reward
        # 生成一个EpisodeStep对象，记录这一步的state和action
        step = EpisodeStep(observation=obs, action=action)
        # 将step添加在episode_steps末尾
        episode_steps.append(step)
        # 如果game is done
        if is_done:
            # 生成一个Episode对象，记录总reward和所有step
            e = Episode(reward=episode_reward, steps=episode_steps)
            # 将本episode对象添加到batch中
            batch.append(e)
            # 重置本episode的reward
            episode_reward = 0.0
            # 重置本episode的所有steps
            episode_steps = []
            # 重置env，获得state
            next_obs = env.reset()
            # 如果批次所有episode均已经完成
            if len(batch) == batch_size:
                # 函数返回batch
                yield batch
                # 清空batch
                batch = []
        # 更新下一步输入的state
        obs = next_obs

# 从所有训练样本中，根据reward，选取比较好的样本来训练网络
def filter_batch(batch, percentile):
    # 将batch中所有episode的reward放在一个list中
    rewards = list(map(lambda s: s.reward, batch))
    # 根据用户设置的percentile，选择入选优质episode的最低reward
    reward_bound = np.percentile(rewards, percentile)
    # 获取所有episode的reward均值，用于判断训练是否完成
    reward_mean = float(np.mean(rewards))

    # 存放优质episode相应的state
    train_obs = []
    # 存储优质episode相应的action
    train_act = []

    # 根据优质episode入选最低reward值，遴选batch中所有优质样本
    for reward, steps in batch:
        # 如果本episode的reward低于入选阈值
        if reward < reward_bound:
            # 跳入下一episode
            continue
        # 如果为优质episode，将本episode每一个step的state添加到train_obs中
        train_obs.extend(map(lambda step: step.observation, steps))
        # 如果为优质episode，将本episode每一个step的action添加到train_obs中
        train_act.extend(map(lambda step: step.action, steps))

    # 将array转换成torch的tensor
    train_obs_v = torch.FloatTensor(train_obs)
    train_act_v = torch.LongTensor(train_act)
    return train_obs_v, train_act_v, reward_bound, reward_mean

# 主程序
if __name__ == "__main__":
    # 生成env
    env = gym.make("CartPole-v0")
    # env = gym.wrappers.Monitor(env, directory="mon", force=True)
    # 获取state space的维度
    obs_size = env.observation_space.shape[0]
    # 获取action space的维度
    n_actions = env.action_space.n

    # 生成神经网络对象
    net = Net(obs_size, HIDDEN_SIZE, n_actions)
    # 定义交叉熵损失函数
    objective = nn.CrossEntropyLoss()
    # 定义优化函数以及学习速率
    optimizer = optim.Adam(params=net.parameters(), lr=0.01)
    # 记录训练过程
    writer = SummaryWriter(comment="-cartpole")

    # 循环批次训练，一批一批直到reward均值满足要求
    for iter_no, batch in enumerate(iterate_batches(
            env, net, BATCH_SIZE)):
        # 筛选优质episode
        obs_v, acts_v, reward_b, reward_m = \
            filter_batch(batch, PERCENTILE)
        # 自动微分清零
        optimizer.zero_grad()
        # 用net获得了action的得分，然后跟优质episode的action（随机）对比
        action_scores_v = net(obs_v)
        # 计算交叉熵损失
        loss_v = objective(action_scores_v, acts_v)
        # 自动微分
        loss_v.backward()
        # 反向传播误差
        optimizer.step()
        # 打印这一批的训练相关信息
        print("%d: loss=%.3f, reward_mean=%.1f, rw_bound=%.1f" % (
            iter_no, loss_v.item(), reward_m, reward_b))
        # 记录训练过程数据，用于tensorboard
        writer.add_scalar("loss", loss_v.item(), iter_no)
        writer.add_scalar("reward_bound", reward_b, iter_no)
        writer.add_scalar("reward_mean", reward_m, iter_no)
        # 当reward均值达到阈值，跳出训练循环
        if reward_m > 199:
            print("Solved!")
            break
    # 关闭记录器
    writer.close()
