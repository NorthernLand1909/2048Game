from tqdm import tqdm
import numpy as np
from core.board import Board
from ai.agent import DQNAgent
from ai.model import DQN

class Trainer:
    def __init__(self, visualizer):
        self.env = Board(load_saved=False)
        self.model = DQN()
        self.agent = DQNAgent(self.model)
        self.episodes = 500  # 增加训练次数
        self.target_update_counter = 0  # 目标网络更新计数器
        self.visualizer = visualizer  # 训练可视化工具

    def train(self):
        progress = tqdm(range(self.episodes))
        for episode in progress:
            self.env = Board(load_saved=False)
            state = self.env.get_normalized_state()
            origin_grid = self.env.get_state()
            total_reward = 0
            done = False
            epsilon = max(0.1, 0.9 - (episode / self.episodes) * 0.8)  # 线性探索衰减
            
            while not done:
                action = self.agent.get_action(origin_grid, state, epsilon)
                prev_score = self.env.score
                moved = self.env.move(action)
                new_state = self.env.get_normalized_state()
                new_origin_grid = self.env.get_state()
                done = self.env.is_game_over()

                reward = self._calculate_reward(prev_score, moved, done)
                total_reward += reward
                self.agent.remember(state, action, reward, new_state, done)
                self.agent.replay()

                self.target_update_counter += 1
                if self.target_update_counter % 1000 == 0:
                    self.agent.target_model.load_state_dict(self.agent.model.state_dict())

                state = new_state
                origin_grid = new_origin_grid

            # 记录数据
            self.visualizer.log_data(episode, self.env.score, total_reward)

            progress.set_description(f"Episode {episode} | Reward: {total_reward:.2f} | Score: {self.env.score}")

        # 训练完成后绘制图表
        self.visualizer.plot_results()

    def _calculate_reward(self, prev_score, moved, done):
        """优化奖励函数"""
        score_gain = 1e-2*np.log2(self.env.score - prev_score + 1) / 10  # 取 log 避免过大
        merge_bonus = 1e-2*np.log2(max(self.env.max_tile(), 2)) / 10  # 奖励更大数值的 tile
        move_penalty = -1e-3 if not moved else 0  # 惩罚无效移动
        gameover_penalty = -1.0 if done else 0  # 失败惩罚

        return score_gain + merge_bonus + move_penalty + gameover_penalty
