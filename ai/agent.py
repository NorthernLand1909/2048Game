import torch
from collections import deque
from utils import DEVICE, BATCH_SIZE, MEMORY_CAPACITY
import random
import numpy as np
from core.board import Board

class DQNAgent:
    def __init__(self, model, lr=2e-2, gamma=0.99):
        self.model = model.to(DEVICE)
        self.target_model = model.__class__().to(DEVICE)
        self.target_model.load_state_dict(self.model.state_dict())
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)
        self.memory = deque(maxlen=MEMORY_CAPACITY)
        self.gamma = gamma
        
    def get_action(self, origin_grid, state, temperature=1.0):
        # 创建临时棋盘并复制当前状态
        temp_board = Board(load_saved=False)
        temp_board.grid = origin_grid.copy()
        valid_moves = temp_board.available_moves()  # 获取有效移动

        # 如果没有有效移动，返回随机方向（避免异常）
        if not valid_moves:
            return random.choice(['Left', 'Right', 'Up', 'Down'])

        # 将状态转换为张量并计算 Q 值
        state_tensor = torch.FloatTensor(state).to(DEVICE)
        with torch.no_grad():
            q_values = self.model(state_tensor).squeeze(0)

        # 仅保留有效移动的 Q 值
        move_to_index = {'Left': 0, 'Right': 1, 'Up': 2, 'Down': 3}
        valid_indices = [move_to_index[move] for move in valid_moves]
        valid_q_values = q_values[valid_indices]

        # 处理可能出现的 NaN / 负数问题
        temperature = max(temperature, 1e-6)  # 避免 temperature 为 0
        valid_q_values = torch.clamp(valid_q_values, min=-1e6, max=1e6)  # 防止极端值
        exp_q = torch.exp(valid_q_values / temperature)
        exp_q = torch.clamp(exp_q, min=1e-6, max=1e6)  # 防止 exp_q 下溢或溢出

        # 归一化为概率分布
        sum_exp_q = torch.sum(exp_q)
        if sum_exp_q == 0:  # 如果 sum_exp_q 为 0，设置均匀分布
            probs = torch.ones_like(exp_q) / len(exp_q)
        else:
            probs = exp_q / sum_exp_q

        # 再次检查 probs 是否有效
        if torch.isnan(probs).any() or torch.sum(probs) == 0:
            print("Invalid probabilities:", probs)
            return random.choice(valid_moves)

        # 根据概率分布采样一个动作
        sampled_index = torch.multinomial(probs, 1).item()
        return valid_moves[sampled_index]  # 确保返回的是合法移动
    
    def remember(self, state, action, reward, next_state, done):
        action_index = ['Left', 'Right', 'Up', 'Down'].index(action)
        self.memory.append((
            torch.FloatTensor(state).to(DEVICE),
            torch.LongTensor([action_index]).to(DEVICE),
            torch.FloatTensor([reward]).to(DEVICE),
            torch.FloatTensor(next_state).to(DEVICE),
            torch.FloatTensor([done]).to(DEVICE),
            abs(reward)  # 存储优先级
        ))
    
    def replay(self):
        if len(self.memory) < BATCH_SIZE:
            return
        
        # 计算经验优先级（确保所有优先级非零）
        priorities = np.array([exp[5] for exp in self.memory]) + 1e-6  

        # 归一化，确保和为1
        probabilities = priorities / np.sum(priorities)

        # 处理可能的 NaN / Inf 计算异常
        probabilities = np.nan_to_num(probabilities, nan=1.0 / len(priorities))

        # 采样优先级高的经验
        sampled_indices = np.random.choice(len(self.memory), BATCH_SIZE, p=probabilities)
        minibatch = [self.memory[i] for i in sampled_indices]

        states, actions, rewards, next_states, dones, _ = zip(*minibatch)
        
        states = torch.stack(states)
        actions = torch.cat(actions)
        rewards = torch.cat(rewards)
        next_states = torch.stack(next_states)
        dones = torch.cat(dones)
        
        current_q = self.model(states).gather(1, actions.unsqueeze(1))
        best_actions = self.model(next_states).argmax(1, keepdim=True)
        next_q = self.target_model(next_states).gather(1, best_actions).squeeze()
        target_q = rewards + (1 - dones) * self.gamma * next_q
        
        # Huber 损失
        loss = torch.nn.functional.smooth_l1_loss(current_q.squeeze(), target_q)
        
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=0.5)
        self.optimizer.step()

        # 软更新目标网络
        tau = 0.01
        for target_param, param in zip(self.target_model.parameters(), self.model.parameters()):
            target_param.data.copy_(tau * param.data + (1 - tau) * target_param.data)
