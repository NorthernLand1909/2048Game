import torch
from tqdm import tqdm
from core.board import Board
from ai.agent import DQNAgent
from ai.model import DQN

from utils import DEVICE

class Trainer:
    def __init__(self):
        self.env = Board(load_saved=False)
        self.model = DQN()
        self.agent = DQNAgent(self.model)
        self.episodes = 100
        
    def train(self):
        progress = tqdm(range(self.episodes))
        for episode in progress:
            self.env = Board(load_saved=False)
            state = self.env.get_normalized_state()
            origin_grid = self.env.get_state()
            total_reward = 0
            done = False
            
            while not done:
                # 获取动作
                action = self.agent.get_action(origin_grid, state, epsilon=(0.2)**episode)
                
                # 执行动作
                prev_score = self.env.score
                moved = self.env.move(action)
                new_state = self.env.get_normalized_state()
                new_origin_grid = self.env.get_state()
                done = self.env.is_game_over()
                
                # 计算奖励
                reward = self._calculate_reward(prev_score, moved, done)
                total_reward += reward
                
                # 存储经验
                self.agent.remember(state, action, reward, new_state, done)
                
                # 训练网络
                self.agent.replay()
                
                state = new_state
                origin_grid = new_origin_grid
                
            progress.set_description(f"Episode {episode} | Reward: {total_reward} | Score: {self.env.score}")
            
    def _calculate_reward(self, prev_score, moved, done):
        """自定义奖励函数"""
        # print(self.env.score, prev_score)
        score_reward = (self.env.score - prev_score) / prev_score if prev_score != 0 else 0
        move_penalty = (1e-8)*prev_score
        gameover_penalty = (-2e4)/prev_score if done else 0
        return score_reward + move_penalty + gameover_penalty