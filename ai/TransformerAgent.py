import random
import torch
import numpy as np
import torch.nn.functional as F

from ai.TransformerModel import TransformerModel
from utils import DEVICE

class TransformerAgent:
    def __init__(self, epsilon=0.1, gamma=0.99, device=DEVICE):
        self.model: TransformerModel = TransformerModel()
        self.model = self.model.to(device)
        self.epsilon = epsilon  # 探索率
        self.gamma = gamma  # 折扣因子
        self.device = device
        self.action_map = {'Left': 0, 'Right': 1, 'Up': 2, 'Down': 3}

    def select_action(self, state, temperature=1.0):
        """
        选择一个动作，完全基于玻尔兹曼策略。
        温度越高，选择越随机；温度越低，选择越集中于最大Q值的动作。
        """
        # 获取当前有效的动作
        valid_moves = self.env.available_moves()  # 使用Board中的available_moves()获取有效动作
        
        # 如果没有有效的移动，随机选择一个动作
        if not valid_moves:
            return random.choice(['Left', 'Right', 'Up', 'Down'])
        
        # 计算当前状态的Q值
        with torch.no_grad():
            # 将状态转换为Tensor
            state_tensor = torch.tensor(state, dtype=torch.float32).to(self.device)
            
            # 计算Q值
            q_values = self.model(state_tensor)
            
            # 将动作映射到索引（可以直接使用q_values的索引）
            move_to_index = {'Left': 0, 'Right': 1, 'Up': 2, 'Down': 3}
            valid_indices = [move_to_index[move] for move in valid_moves]
            valid_q_values = q_values[0, valid_indices]  # 获取有效动作的Q值

            # 防止Q值过小或过大，避免溢出问题
            temperature = max(temperature, 1e-6)  # 避免温度为0
            valid_q_values = torch.clamp(valid_q_values, min=-1e6, max=1e6)  # 限制极端值
            
            # 计算指数函数，得到每个有效动作的概率分布
            exp_q = torch.exp(valid_q_values / temperature)
            exp_q = torch.clamp(exp_q, min=1e-6, max=1e6)  # 防止下溢或溢出

            # 归一化概率
            sum_exp_q = torch.sum(exp_q)
            if sum_exp_q == 0:
                probs = torch.ones_like(exp_q) / len(exp_q)  # 如果总和为0，设定均匀分布
            else:
                probs = exp_q / sum_exp_q

            # 采样一个动作
            sampled_index = torch.multinomial(probs, 1).item()
            action_str = valid_moves[sampled_index]

        return action_str  # 直接返回有效动作的字符串
    
    def learn(self, state, next_state, action_idx, reward, done, optimizer, delta=1.0):
        """
        使用经验回放来更新Q值。
        """
        # 将np数组转换为torch tensor
        # 如果 state 已经是 Tensor，直接 clone().detach() 以避免 PyTorch 警告
        if isinstance(state, torch.Tensor):
            state_tensor = state.clone().detach().to(self.device)
        else:
            state_tensor = torch.tensor(state, dtype=torch.float32).to(self.device)

        # 如果 next_state 已经是 Tensor
        if isinstance(next_state, torch.Tensor):
            next_state_tensor = next_state.clone().detach().to(self.device)
        else:
            next_state_tensor = torch.tensor(next_state, dtype=torch.float32).to(self.device)

        # 计算当前状态的Q值
        q_values = self.model(state_tensor)
        q_value = q_values[0, action_idx]  # 选择action的Q值
        
        # 计算目标Q值
        with torch.no_grad():
            next_q_values = self.model(next_state_tensor)
            next_q_value = torch.max(next_q_values, dim=1)[0]  # 最大Q值
        
        # 计算目标
        target_q_value = reward + (1 - done) * self.gamma * next_q_value
        
        # 确保 q_value 和 target_q_value 都是标量张量或形状为 (1,)
        q_value = q_value.unsqueeze(0)  # 转换为形状 (1,)
        target_q_value = target_q_value.unsqueeze(0)  # 转换为形状 (1,)

        # 计算 Huber 损失
        loss = F.smooth_l1_loss(q_value, target_q_value, beta=delta)
        
        # 反向传播和优化
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        return loss.item()