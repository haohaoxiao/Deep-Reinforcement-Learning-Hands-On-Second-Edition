import gym

# 进入主程序
if __name__ == "__main__":
    # 创建env对象
    env = gym.make("CartPole-v0")

    # 初始化总reward
    total_reward = 0.0
    # 初始化总步数
    total_steps = 0
    # 初始化state
    obs = env.reset()

    while True:
        # 随机生成一个action
        action = env.action_space.sample()
        # 根据action，通过step获得state，reward和done
        obs, reward, done, _ = env.step(action)
        # 更新总reward
        total_reward += reward
        # 更新总步数
        total_steps += 1
        # 根据game是否done决定是否结束循环
        if done:
            break

    # 输出总成绩
    print("Episode done in %d steps, total reward %.2f" % (
        total_steps, total_reward))
