import torch
import torch.nn as nn
from utils import DEVICE, GRID_SIZE

class DQN(nn.Module):
    def __init__(self):
        super(DQN, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(1, 128, kernel_size=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(128, 256, kernel_size=2, padding=1),
            nn.ReLU()
        )
        self.fc = nn.Sequential(
            nn.Linear(256*(GRID_SIZE+2)*(GRID_SIZE+2), 512),
            nn.ReLU(),
            nn.Linear(512, 4)  # 输出4个动作的Q值
        )
        
    def forward(self, x):
        x = self.conv(x)
        x = torch.flatten(x, -3)
        x = self.fc(x)
        return x

class TransformerModel(nn.Module):
    """备选Transformer模型"""
    def __init__(self):
        super().__init__()
        self.embedding = nn.Embedding(16, 64)  # 2^16=65536
        encoder_layer = nn.TransformerEncoderLayer(d_model=64, nhead=8)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=6)
        self.fc = nn.Linear(64*GRID_SIZE*GRID_SIZE, 4)
        
    def forward(self, x):
        x = self.embedding(x.long())
        x = x.view(GRID_SIZE*GRID_SIZE, -1, 64)
        x = self.transformer(x)
        x = x.view(x.size(0), -1)
        return self.fc(x)