import torch
from collections import deque
from utils import DEVICE, BATCH_SIZE, MEMORY_CAPACITY
import random
from core.board import Board

class DQNAgent:
    def __init__(self, model, lr=1e-5, gamma=0.99):
        self.model = model.to(DEVICE)
        self.target_model = model.__class__().to(DEVICE)
        self.target_model.load_state_dict(self.model.state_dict())
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)
        self.memory = deque(maxlen=MEMORY_CAPACITY)
        self.gamma = gamma
        
    def get_action(self, origin_grid, state, epsilon=0.1):
        temp_board = Board(load_saved=False)
        temp_board.grid = origin_grid.copy()
        valid_moves = temp_board.available_moves()

        if random.random() < epsilon:
            return random.choice(valid_moves)
        
        state_tensor = torch.FloatTensor(state).to(DEVICE)
        with torch.no_grad():
            q_values = self.model(state_tensor)

        move_scores = {
            'Left': q_values[0],
            'Right': q_values[1],
            'Up': q_values[2],
            'Down': q_values[3]
        }
        # 只考虑有效移动的分数
        valid_scores = {move: move_scores[move] for move in valid_moves}
        final_move = max(valid_scores, key=valid_scores.get)

        return final_move
    
    def remember(self, state, action, reward, next_state, done):
        action_index = ['Left', 'Right', 'Up', 'Down'].index(action)
        self.memory.append((
            torch.FloatTensor(state).to(DEVICE),
            torch.LongTensor([action_index]).to(DEVICE),
            torch.FloatTensor([reward]).to(DEVICE),
            torch.FloatTensor(next_state).to(DEVICE),
            torch.FloatTensor([done]).to(DEVICE)
        ))
    
    def replay(self):
        if len(self.memory) < BATCH_SIZE:
            return
        
        batch = random.sample(self.memory, BATCH_SIZE)
        states, actions, rewards, next_states, dones = zip(*batch)
        
        states = torch.stack(states)
        actions = torch.cat(actions)
        rewards = torch.cat(rewards)
        next_states = torch.stack(next_states)
        dones = torch.cat(dones)
        
        current_q = self.model(states).gather(1, actions.unsqueeze(1))
        best_actions = self.model(next_states).argmax(1, keepdim=True)
        next_q = self.target_model(next_states).gather(1, best_actions).squeeze()
        target_q = rewards + (1 - dones) * self.gamma * next_q
        
        # 使用Huber损失（smooth_l1_loss）计算损失
        loss = torch.nn.functional.smooth_l1_loss(current_q.squeeze(), target_q)
        
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=0.5)
        self.optimizer.step()
        
        # 更新目标网络
        tau = 0.01  # 软更新系数
        for target_param, param in zip(self.target_model.parameters(), self.model.parameters()):
            target_param.data.copy_(tau * param.data + (1 - tau) * target_param.data)
