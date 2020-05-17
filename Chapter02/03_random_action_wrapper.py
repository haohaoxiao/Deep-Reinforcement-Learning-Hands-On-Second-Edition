# 本程序演示了gym包里面的wrapper功能来扩展
# 默认gym环境的state，action和reward。本
# 例中演示了如何扩展action功能

import gym
# 类型注释
from typing import TypeVar
import random

# 自定义一个action类型注释
Action = TypeVar('Action')

# 自定义action类，继承自actionwrapper
class RandomActionWrapper(gym.ActionWrapper):
    # 构造函数，添加一个成员变量epsilon
    def __init__(self, env, epsilon=0.1):
        super(RandomActionWrapper, self).__init__(env)
        self.epsilon = epsilon

    # 输入一个action，然后随机判断是返回输入的action还是随机给
    # 出一个action。这个过程自定义了原来的action函数
    def action(self, action: Action) -> Action:
        if random.random() < self.epsilon:
            print("Random!")
            return self.env.action_space.sample()
        return action

# 进入主程序
if __name__ == "__main__":
    # 生成env对象
    env = RandomActionWrapper(gym.make("CartPole-v0"))

    # 初始化state和总reward
    obs = env.reset()
    total_reward = 0.0

    while True:
        # 获得state, reward, done状态
        obs, reward, done, _ = env.step(0)
        # 更新总reward
        total_reward += reward
        # 判断game是否done
        if done:
            break

    # 输出总reward
    print("Reward got: %.2f" % total_reward)
