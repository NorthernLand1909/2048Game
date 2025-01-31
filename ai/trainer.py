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
            state = self.env.get_normalization_state()
            total_reward = 0
            done = False
            
            while not done:
                # 获取动作
                action = self.agent.get_action(state, epsilon=0.2)
                
                # 执行动作
                prev_score = self.env.score
                moved = self.env.move(action)
                new_state = self.env.get_normalization_state()
                done = self.env.is_game_over()
                
                # 计算奖励
                reward = self._calculate_reward(prev_score, moved, done)
                total_reward += reward
                
                # 存储经验
                self.agent.remember(state, action, reward, new_state, done)
                
                # 训练网络
                self.agent.replay()
                
                state = new_state
                
            progress.set_description(f"Episode {episode} | Reward: {total_reward}")
            
    def _calculate_reward(self, prev_score, moved, done):
        """自定义奖励函数"""
        score_reward = (self.env.score - prev_score) / 100.0
        move_penalty = -0.1 if not moved else 0
        gameover_penalty = -10 if done else 0
        return score_reward + move_penalty + gameover_penalty