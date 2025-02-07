import random
import torch
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt

from core.board import Board
from utils import BATCH_SIZE
from ai.Transformer.TransformerAgent import TransformerAgent
from ai.GradientMonitor import GradientMonitor  # 确保导入梯度监控类
from ai.TrainingVisualizer import TrainingVisualizer  # 确保导入可视化类

class TransformerTrainer:
    def __init__(self, batch_size=BATCH_SIZE, buffer_size=10000, learning_rate=1e-4, log_interval=100):
        self.agent: TransformerAgent = TransformerAgent()
        self.env: Board = Board(load_saved=False)
        self.batch_size = batch_size
        self.buffer_size = buffer_size
        self.learning_rate = learning_rate
        
        # 将环境传递给 agent
        self.agent.env = self.env
        
        # 初始化优化器
        self.optimizer = optim.Adam(self.agent.model.parameters(), lr=self.learning_rate)
        
        # 初始化经验回放
        self.memory = []

        # 添加梯度监视器
        self.gradient_monitor = GradientMonitor(self.agent.model, log_interval=log_interval)

        # 添加训练可视化工具
        self.visualizer = TrainingVisualizer()

    def store_experience(self, state, next_state, action, reward, done):
        """ 存储经验 """
        if len(self.memory) >= self.buffer_size:
            self.memory.pop(0)  # 如果缓冲区已满，移除最旧的经验
        self.memory.append((state, next_state, action, reward, done))
    
    def _calculate_reward(self, prev_score, prev_max_tile, moved, done):
        """
        计算奖励
        """
        score_diff = self.env.score - prev_score
        score_gain = np.log1p(score_diff) * 0.001  
        max_tile = self.env.max_tile()  
        merge_bonus = np.log1p(max_tile) * 0.03  

        if max_tile > prev_max_tile:
            merge_bonus += np.log1p(max_tile) * 0.05  

        move_penalty = -0.007  
        
        if done:
            if max_tile < 1024:
                gameover_penalty = -5.0 / np.sqrt(max_tile + 1)  
            else:
                gameover_penalty = -2.0 / np.log1p(max_tile)  
        else:
            gameover_penalty = 0

        return score_gain + merge_bonus + move_penalty + gameover_penalty
    
    def train(self, num_episodes=2000):
        """ 训练过程 """
        step = 0  # 记录全局训练步数
        for episode in range(num_episodes):
            # 重置游戏环境
            self.env = Board(load_saved=False)
            state = self.env.get_normalized_state(model='transformer')  
            prev_score = 0  
            prev_max_tile = 0  
            done = False
            episode_reward = 0
            
            while not done:
                # 使用agent选择一个动作（返回的是字符串）
                temperature = max(1e-4, 1.0 * (0.99 ** episode))
                action_str = self.agent.select_action(state, temperature)
                
                # 在环境中执行动作
                self.env.move(action_str)
                
                # 获取新状态和奖励
                next_state = self.env.get_normalized_state(model='transformer')
                moved = state != next_state  
                reward = self._calculate_reward(prev_score, prev_max_tile, moved, self.env.is_game_over())
                
                # 更新最大方块和当前得分
                prev_max_tile = self.env.max_tile()
                prev_score = self.env.score
                
                # 游戏是否结束
                done = self.env.is_game_over()
                
                # 存储经验
                self.store_experience(state, next_state, action_str, reward, done)
                
                # 经验回放训练
                if len(self.memory) >= self.batch_size:
                    batch = random.sample(self.memory, self.batch_size)
                    states, next_states, actions, rewards, dones = zip(*batch)
                    states = torch.from_numpy(np.array(states, dtype=np.float32)).to(self.agent.device)
                    next_states = torch.from_numpy(np.array(next_states, dtype=np.float32)).to(self.agent.device)
                    actions = torch.tensor([self.agent.action_map[a] for a in actions], dtype=torch.long).to(self.agent.device)
                    rewards = torch.tensor(rewards, dtype=torch.float32).to(self.agent.device)
                    dones = torch.tensor(dones, dtype=torch.float32).to(self.agent.device)

                    # 调用 agent 的 learn 方法进行批量训练
                    loss = self.agent.learn(states, next_states, actions, rewards, dones, self.optimizer)

                    # 记录梯度
                    self.gradient_monitor.log_gradients(step)
                    step += 1
                
                # 更新状态
                state = next_state
                episode_reward += reward

                if step % 1000 == 0:
                    print(f"Step {step}, epi_reward: {episode_reward}, epi_score: {self.env.score}")
            
            # 记录训练数据
            self.visualizer.log_data(episode, self.env.score, episode_reward)

            print(f"Episode {episode}, Reward: {episode_reward}, Score: {self.env.score}, temperature: {temperature:.4f}")

        # 训练结束后可视化结果
        self.visualizer.plot_results()
