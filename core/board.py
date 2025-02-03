import numpy as np
from utils import GRID_SIZE, NEW_TILE, SAVE_FILE
import json
import os
import random

class Board:
    def __init__(self, load_saved=True):  # 新增load_saved参数
        self.grid = np.zeros((GRID_SIZE, GRID_SIZE), dtype=int)
        self.previous_grid = None
        self.score = 0
        self.high_score = 0
        
        # 只有在需要时才加载存档
        if load_saved:
            self.load_game()
        
        # 如果是全新游戏且棋盘为空，生成初始方块
        if self.is_empty():
            self.spawn_tile()
            self.spawn_tile()

    def is_empty(self):
        for row in self.grid:
            if any(row):
                return False
        return True

    def spawn_tile(self):
        empty_cells = [(r, c) for r in range(GRID_SIZE) for c in range(GRID_SIZE) if self.grid[r][c] == 0]
        if empty_cells:
            r, c = random.choice(empty_cells)
            self.grid[r][c] = NEW_TILE
            return (r, c)
        return None

    def save_state(self):
        self.previous_grid = np.copy(self.grid)

    def undo(self):
        if self.previous_grid:
            self.grid = np.copy(self.previous_grid)
            self.previous_grid = None

    def save_game(self):
        data = {
            "grid": self.grid.tolist(),
            "score": int(self.score),
            "high_score": int(self.high_score)
        }
        with open(SAVE_FILE, "w") as file:
            json.dump(data, file)

    def load_game(self):
        if os.path.exists(SAVE_FILE):
            try:
                with open(SAVE_FILE, "r") as file:
                    data = json.load(file)
                    loaded_grid = data.get("grid", [])
                    if len(loaded_grid) == GRID_SIZE and all(len(row) == GRID_SIZE for row in loaded_grid):
                        self.grid = np.array(loaded_grid).copy()
                        self.score = data.get("score", 0)
                        self.high_score = data.get("high_score", 0)
            except (json.JSONDecodeError, KeyError):
                pass

    def move(self, direction):
        self.save_state()
        moved = False
        new_score = 0

        def compress(row):
            # 使用numpy的数组操作
            new_row = row[row != 0]
            new_row = np.concatenate([new_row, np.zeros(GRID_SIZE - len(new_row), dtype=int)])
            return new_row

        def merge(row):
            nonlocal new_score
            new_row = compress(row)
            for i in range(GRID_SIZE - 1):
                if new_row[i] != 0 and new_row[i] == new_row[i + 1]:
                    new_row[i] *= 2
                    new_score += new_row[i]
                    new_row[i + 1] = 0
            return compress(new_row)

        for i in range(GRID_SIZE):
            if direction in ('Left', 'Right'):
                row = self.grid[i, :]
                if direction == 'Right':
                    row = row[::-1]  # reverse the row using slicing
                row = merge(row)
                if direction == 'Right':
                    row = row[::-1]  # reverse the row back
                if not np.array_equal(row, self.grid[i, :]):
                    moved = True
                self.grid[i, :] = row
            else:
                col = self.grid[:, i]
                if direction == 'Down':
                    col = col[::-1]  # reverse the column using slicing
                col = merge(col)
                if direction == 'Down':
                    col = col[::-1]  # reverse the column back
                original_col = self.grid[:, i]
                if not np.array_equal(col, original_col):
                    moved = True
                self.grid[:, i] = col

        if moved:
            self.score += new_score
            if self.score > self.high_score:
                self.high_score = self.score
            self.spawn_tile()
            return True
        else:
            return False
    
    def max_tile(self):
        return np.max(self.grid)

    def get_state(self):
        return self.grid
    
    def get_normalized_state(self, model='DQN'):
        grid_with_no_zeros = np.where(self.grid == 0, 1, self.grid)
        log_state = np.log2(grid_with_no_zeros.astype(np.float32))
        log_min = log_state.min()
        log_max = log_state.max()
        
        if log_max - log_min == 0:
            normalized_state = np.zeros_like(log_state)
        else:
            normalized_state = (log_state - log_min) / (log_max - log_min)

        if model == 'DQN':
            normalized_state = np.expand_dims(normalized_state, axis=0)
        elif model == 'transformer':
            # 将状态展平成一维数组，确保符合模型输入的要求
            normalized_state = normalized_state.flatten()  # 从4x4矩阵展平为16维的向量
            normalized_state = np.expand_dims(normalized_state, axis=0)  # 扩展为(1, 16)
        return normalized_state
    
    def available_moves(self):
        """返回有效移动方向"""
        valid_moves = []
        for direction in ['Left', 'Right', 'Up', 'Down']:
            if self.check_move_valid(direction):
                valid_moves.append(direction)
        return valid_moves

    def check_move_valid(self, direction):
        """优化后的移动有效性检查"""
        temp_board = Board(load_saved=False)
        temp_board.grid = self.grid.copy()
        return temp_board.move(direction)

    def is_game_over(self):
        """更精确的游戏结束检测"""
        # 检查是否有空单元格
        if np.any(self.grid == 0):
            return False
            
        # 检查所有可能的合并方向
        for direction in ['Left', 'Right', 'Up', 'Down']:
            if self.check_move_valid(direction):
                return False
        return True
