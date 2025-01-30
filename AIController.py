import random

class AIController:
    def get_move(self, state):
        # 示例AI随机选择方向
        return random.choice(["Left", "Right", "Up", "Down"])