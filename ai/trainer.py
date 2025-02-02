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
        self.episodes = 1000  # 增加训练次数
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
                prev_max_tile = self.env.max_tile()
                moved = self.env.move(action)
                new_state = self.env.get_normalized_state()
                new_origin_grid = self.env.get_state()
                done = self.env.is_game_over()

                reward = self._calculate_reward(prev_score, prev_max_tile, moved, done)
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

    def _calculate_reward(self, prev_score, prev_max_tile, moved, done):
        """优化奖励函数"""
        score_diff = self.env.score - prev_score
        
        # 分数奖励（适中）
        score_gain = np.log1p(score_diff) * 0.01  # log1p 避免 log(0) 问题
        
        # 合成奖励（比分数奖励略高）
        max_tile = self.env.max_tile()
        merge_bonus = np.log1p(max_tile) * 0.015  # 保证合成奖励略高
        
        # 首次合成更大数值时额外奖励
        if max_tile > prev_max_tile:
            merge_bonus += np.log1p(max_tile) * 0.02  # 额外奖励，增益幅度稍大
            prev_max_tile = max_tile  # 记录历史最大 tile
        
        # 移动惩罚（较小）
        move_penalty = -0.0005 if not moved else 0  # 轻微惩罚无效移动
        
        # 失败惩罚（随总分增加而减少）
        gameover_penalty = -1.0 / np.log1p(self.env.score) if done else 0  
        
        # 组合最终奖励
        reward = score_gain + merge_bonus + move_penalty + gameover_penalty
        
        return reward
