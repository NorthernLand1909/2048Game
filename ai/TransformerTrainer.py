import random
import torch
import torch.optim as optim
import numpy as np

from core.board import Board

from ai.TransformerAgent import TransformerAgent

class TransformerTrainer:
    def __init__(self, batch_size=128, buffer_size=10000, learning_rate=1e-4):
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
        
    def store_experience(self, state, next_state, action, reward, done):
        """ 存储经验 """
        if len(self.memory) >= self.buffer_size:
            self.memory.pop(0)  # 如果缓冲区已满，移除最旧的经验
        self.memory.append((state, next_state, action, reward, done))
    
    def _calculate_reward(self, prev_score, prev_max_tile, moved, done):
        """
        更细致准确的奖励函数，计算当前奖励
        """
        score_diff = self.env.score - prev_score
        
        # 进一步降低得分奖励的权重
        score_gain = np.log1p(score_diff) * 0.001  # score gain in log scale
        
        # 调整合成奖励，使其仍远超得分奖励但不会过大
        max_tile = self.env.max_tile()  # 获取当前最大方块
        merge_bonus = np.log1p(max_tile) * 0.03  # 基于最大方块进行奖励

        # 调整首次合成更大数值的奖励，避免数值过大
        if max_tile > prev_max_tile:
            merge_bonus += np.log1p(max_tile) * 0.05  # 如果是首次合成更大数值的方块，给出额外奖励
        
        # 轻微增加移动惩罚，避免频繁无效移动
        move_penalty = -0.007  # 每次移动的轻微惩罚
        
        # 游戏结束时，根据最大块给予非线性惩罚
        if done:
            if max_tile < 1024:
                gameover_penalty = -5.0 / np.sqrt(max_tile + 1)  # 块越小，惩罚越大
            else:
                gameover_penalty = -2.0 / np.log1p(max_tile)  # 1024以上，惩罚平缓
        else:
            gameover_penalty = 0

        # 综合所有奖励与惩罚
        return score_gain + merge_bonus + move_penalty + gameover_penalty
    
    def train(self, num_episodes=1000):
        """ 训练过程 """
        for episode in range(num_episodes):
            # 重置游戏环境
            self.env = Board(load_saved=False)
            state = self.env.get_normalized_state(model='transformer')  # 获取标准化后的状态
            prev_score = 0  # 初始得分
            prev_max_tile = 0  # 初始最大方块
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
                moved = state != next_state  # 检查是否有移动
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
                    # 将所有批次数据传递给 agent
                    states, next_states, actions, rewards, dones = zip(*batch)
                    states = torch.from_numpy(np.array(states, dtype=np.float32)).to(self.agent.device)
                    next_states = torch.from_numpy(np.array(next_states, dtype=np.float32)).to(self.agent.device)
                    actions = torch.tensor([self.agent.action_map[a] for a in actions], dtype=torch.long).to(self.agent.device)
                    rewards = torch.tensor(rewards, dtype=torch.float32).to(self.agent.device)
                    dones = torch.tensor(dones, dtype=torch.float32).to(self.agent.device)

                    # 调用 agent 的 learn 方法进行批量训练
                    loss = self.agent.learn(states, next_states, actions, rewards, dones, self.optimizer)
                
                # 更新状态
                state = next_state
                episode_reward += reward
            
            print(f"Episode {episode}, Reward: {episode_reward}, Score: {self.env.score}, temperature: {temperature:.4f}")
