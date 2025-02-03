from utils import DEVICE
from core.board import Board
from ai.model import DQN
from ai.TransformerModel import TransformerModel
import torch

class TrainedAgent:
    def __init__(self, model):
        if model == 'DQN':
            self.model = DQN().to(DEVICE)
            self.model.load_state_dict(torch.load('2048_dqn.pth'))
        elif model == 'transformer':
            self.model = TransformerModel().to(DEVICE)
            self.model.load_state_dict(torch.load('2048_transformer.pth'))
        self.model.eval()
    
    def get_move(self, origin_grid, normalized_grid):
        # 获取有效移动方向
        valid_moves = self._get_valid_moves(origin_grid)
        if not valid_moves:
            return None  # 游戏结束

        # 获取神经网络预测的Q值
        q_values = self._predict_q_values(normalized_grid)
        
        # 选择有效移动中Q值最大的方向
        return self._select_best_valid_move(q_values, valid_moves)

    def _get_valid_moves(self, grid):
        """获取当前状态的有效移动方向"""
        temp_board = Board(load_saved=False)
        temp_board.grid = grid.copy()
        return temp_board.available_moves()

    def _predict_q_values(self, state):
        """获取神经网络预测的Q值"""
        state_tensor = torch.FloatTensor(state).to(DEVICE)
        with torch.no_grad():
            return self.model(state_tensor).cpu().numpy().flatten()

    def _select_best_valid_move(self, q_values, valid_moves):
        """在有效移动中选择最优方向"""
        move_scores = {
            'Left': q_values[0],
            'Right': q_values[1],
            'Up': q_values[2],
            'Down': q_values[3]
        }
        # 只考虑有效移动的分数
        valid_scores = {move: move_scores[move] for move in valid_moves}
        final_move = max(valid_scores, key=valid_scores.get)
        print(f"qval: {q_values}, AI move: {final_move}")
        return final_move
    
import random
class RandomAgent:
    """随机选择方向的AI代理"""
    def __init__(self):
        pass
    def get_move(self, origin_grid, normalized_grid):
        return random.choice(["Left", "Right", "Up", "Down"])