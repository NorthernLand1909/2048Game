import torch

# 游戏配置
GRID_SIZE = 4
NEW_TILE = 2
SAVE_FILE = "2048_save.json"
TILE_COLORS = {
    0: "#CDC1B4", 2: "#EEE4DA", 4: "#EDE0C8", 8: "#F2B179",
    16: "#F59563", 32: "#F67C5F", 64: "#F65E3B", 128: "#EDCF72",
    256: "#EDCC61", 512: "#EDC850", 1024: "#EDC53F", 2048: "#EDC22E"
}

# 设备配置
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
BATCH_SIZE = 128
MEMORY_CAPACITY = 10000