import torch
import torch.nn as nn

from utils import GRID_SIZE

class TransformerModel(nn.Module):
    def __init__(self, input_size=GRID_SIZE*GRID_SIZE, hidden_dim=128, num_heads=4, num_layers=2, output_size=4):
        super(TransformerModel, self).__init__()
        
        # 位置编码（Position Encoding）
        self.position_encoding = nn.Parameter(torch.randn(1, input_size, hidden_dim))
        
        # Transformer编码器层
        self.transformer_encoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model=hidden_dim, nhead=num_heads, batch_first=True),  # 设置 batch_first=True
            num_layers=num_layers
        )

        
        # 输入层，将每个格子映射到隐藏空间
        self.input_layer = nn.Linear(input_size, hidden_dim)
        
        # 输出层，生成Q值
        self.output_layer = nn.Linear(hidden_dim, output_size)

    def forward(self, x):
        # 输入数据处理
        x = self.input_layer(x)  # 输入层
        x = x + self.position_encoding  # 加上位置编码
        
        # Transformer Encoder
        x = x.transpose(0, 1)  # Transformer expects (seq_len, batch_size, features)
        x = self.transformer_encoder(x)
        
        # 取Transformer输出的最后一个token
        x = x[-1, :, :]
        
        # 输出Q值
        q_values = self.output_layer(x)
        
        return q_values
