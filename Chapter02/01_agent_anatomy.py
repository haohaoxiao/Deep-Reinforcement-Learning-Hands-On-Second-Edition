import random
# 类型注释
from typing import List

# Environment类定义
class Environment:

    # 构造函数
    def __init__(self):
        # game还剩多少步
        self.steps_left = 10

    # 获得state
    def get_observation(self) -> List[float]:
        return [0.0, 0.0, 0.0]

    # 获得action的样本库
    def get_actions(self) -> List[int]:
        return [0, 1]

    # 判断game结束否
    def is_done(self) -> bool:
        # 当game剩余步数为0
        return self.steps_left == 0

    # 根据action获得reward
    def action(self, action: int) -> float:
        # 先判断game是否结束
        if self.is_done():
            raise Exception("Game is over")
        # 剩余步数减1
        self.steps_left -= 1
        # 随机返回一个值作为reward
        return random.random()

# Agent类定义
class Agent:

    # 构造函数
    def __init__(self):
        # 初始化reward
        self.total_reward = 0.0

    # Agent根据环境给出action和reward
    def step(self, env: Environment):
        # 获取states
        current_obs = env.get_observation()
        # 获取action样本库
        actions = env.get_actions()
        # 获取本步骤reward
        reward = env.action(random.choice(actions))
        # 更新总reward
        self.total_reward += reward

# 进入主程序
if __name__ == "__main__":
    
    # 生成Environment对象
    env = Environment()

    # 生成Agent对象
    agent = Agent()

    # 从env判断game是否结束
    while not env.is_done():
        # agent根据env作出一个action
        agent.step(env)

    # 输出总reward
    print("Total reward got: %.4f" % agent.total_reward)
