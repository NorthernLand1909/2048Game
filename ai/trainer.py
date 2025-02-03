from tqdm import tqdm
import numpy as np
from core.board import Board
from ai.agent import DQNAgent
from ai.model import DQN
from ai.GradientMonitor import GradientMonitor  # 添加这一行

class Trainer:
    def __init__(self, visualizer):
        self.env = Board(load_saved=False)
        self.model = DQN()
        self.agent = DQNAgent(self.model)
        self.episodes = 1000  # 训练轮数
        self.target_update_counter = 0
        self.visualizer = visualizer
        self.grad_monitor = GradientMonitor(self.model, log_interval=2000, verbose=True)  # 添加梯度监测

    def train(self):
        progress = tqdm(range(self.episodes))
        for episode in progress:
            self.env = Board(load_saved=False)
            state = self.env.get_normalized_state()
            origin_grid = self.env.get_state()
            total_reward = 0
            done = False

            # Boltzmann 采样的温度（训练前期探索多，后期稳定）
            temperature = max(1e-4, 1.0 * (0.99 ** episode))

            while not done:
                action = self.agent.get_action(origin_grid, state, temperature)
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

                # 记录梯度
                self.grad_monitor.log_gradients(self.target_update_counter)

                self.target_update_counter += 1
                if self.target_update_counter % 1000 == 0:
                    self.agent.target_model.load_state_dict(self.agent.model.state_dict())

                state = new_state
                origin_grid = new_origin_grid

            self.visualizer.log_data(episode, self.env.score, total_reward)
            progress.set_description(f"Episode {episode} | Reward: {total_reward:.2f} | Score: {self.env.score}")

        self.visualizer.plot_results()

    def _calculate_reward(self, prev_score, prev_max_tile, moved, done):
        score_diff = self.env.score - prev_score

        # 进一步降低得分奖励的权重
        score_gain = np.log1p(score_diff) * 0.001  

        # 调整合成奖励，使其仍远超得分奖励但不会过大
        max_tile = self.env.max_tile()
        merge_bonus = np.log1p(max_tile) * 0.03  

        # 调整首次合成更大数值的奖励，避免数值过大
        if max_tile > prev_max_tile:
            merge_bonus += np.log1p(max_tile) * 0.05  

        # 轻微增加移动惩罚，避免频繁无效移动
        move_penalty = -0.007  

        # 游戏结束时，根据最大块给予非线性惩罚
        if done:
            if max_tile < 1024:
                gameover_penalty = -5.0 / np.sqrt(max_tile + 1)  # 块越小，惩罚越大
            else:
                gameover_penalty = -2.0 / np.log1p(max_tile)  # 1024以上，惩罚平缓
        else:
            gameover_penalty = 0

        return score_gain + merge_bonus + move_penalty + gameover_penalty
