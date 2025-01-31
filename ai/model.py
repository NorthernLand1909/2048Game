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

        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
        
    def forward(self, x):
        x = self.conv(x)
        x = torch.flatten(x, -3)
        x = self.fc(x)
        return x
