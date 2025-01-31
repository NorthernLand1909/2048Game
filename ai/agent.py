import numpy as np
import torch
from collections import deque
from utils import DEVICE, BATCH_SIZE, MEMORY_CAPACITY
from ai.model import DQN
import random

class DQNAgent:
    def __init__(self, model, lr=1e-4, gamma=0.99):
        self.model = model.to(DEVICE)
        self.target_model = model.__class__().to(DEVICE)
        self.target_model.load_state_dict(self.model.state_dict())
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)
        self.memory = deque(maxlen=MEMORY_CAPACITY)
        self.gamma = gamma
        
    def get_action(self, state, epsilon=0.1):
        if random.random() < epsilon:
            return random.choice(['Left', 'Right', 'Up', 'Down'])
        
        state_tensor = torch.FloatTensor(state).to(DEVICE)
        with torch.no_grad():
            q_values = self.model(state_tensor)
        return ['Left', 'Right', 'Up', 'Down'][torch.argmax(q_values).item()]
    
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
        next_q = self.target_model(next_states).max(1)[0].detach()
        target_q = rewards + (1 - dones) * self.gamma * next_q
        
        loss = torch.nn.functional.mse_loss(current_q.squeeze(), target_q)
        
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
        # 更新目标网络
        self.target_model.load_state_dict(self.model.state_dict())
