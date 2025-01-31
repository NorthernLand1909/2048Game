from matplotlib import pyplot as plt

class TrainingVisualizer:
    def __init__(self):
        self.episodes = []
        self.scores = []
        self.rewards = []

    def log_data(self, episode, score, reward):
        """记录训练过程中每一轮的分数和奖励"""
        self.episodes.append(episode)
        self.scores.append(score)
        self.rewards.append(reward)

    def plot_results(self):
        """绘制训练结果"""
        plt.figure(figsize=(12, 5))

        # 绘制分数曲线
        plt.subplot(1, 2, 1)
        plt.plot(self.episodes, self.scores, label="Score", color="blue")
        plt.xlabel("Episode")
        plt.ylabel("Score")
        plt.title("Training Score over Episodes")
        plt.legend()

        # 绘制奖励曲线
        plt.subplot(1, 2, 2)
        plt.plot(self.episodes, self.rewards, label="Reward", color="green")
        plt.xlabel("Episode")
        plt.ylabel("Total Reward")
        plt.title("Training Reward over Episodes")
        plt.legend()

        plt.show()