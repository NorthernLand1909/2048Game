import numpy as np
from utils import GRID_SIZE, NEW_TILE, SAVE_FILE
import json
import os
import random

class Board:
    def __init__(self, load_saved=True):  # 新增load_saved参数
        self.grid = [[0] * GRID_SIZE for _ in range(GRID_SIZE)]
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
        self.previous_grid = [row[:] for row in self.grid]

    def undo(self):
        if self.previous_grid:
            self.grid = [row[:] for row in self.previous_grid]
            self.previous_grid = None

    def save_game(self):
        data = {
            "grid": self.grid,
            "score": self.score,
            "high_score": self.high_score
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
                        self.grid = loaded_grid
                        self.score = data.get("score", 0)
                        self.high_score = data.get("high_score", 0)
            except (json.JSONDecodeError, KeyError):
                pass

    def move(self, direction):
        self.save_state()
        moved = False
        new_score = 0

        def compress(row):
            new_row = [v for v in row if v != 0]
            new_row += [0] * (GRID_SIZE - len(new_row))
            return new_row

        def merge(row):
            nonlocal new_score
            new_row = compress(row)
            for i in range(GRID_SIZE - 1):
                if new_row[i] and new_row[i] == new_row[i + 1]:
                    new_row[i] *= 2
                    new_score += new_row[i]
                    new_row[i + 1] = 0
            return compress(new_row)

        for i in range(GRID_SIZE):
            if direction in ('Left', 'Right'):
                row = self.grid[i][:]
                if direction == 'Right':
                    row.reverse()
                row = merge(row)
                if direction == 'Right':
                    row.reverse()
                if row != self.grid[i]:
                    moved = True
                self.grid[i] = row
            else:
                # 修正这里：self.board -> self
                col = [self.grid[r][i] for r in range(GRID_SIZE)]  # 修正点
                if direction == 'Down':
                    col.reverse()
                col = merge(col)
                if direction == 'Down':
                    col.reverse()
                original_col = [self.grid[r][i] for r in range(GRID_SIZE)]
                if col != original_col:
                    moved = True
                for r in range(GRID_SIZE):
                    self.grid[r][i] = col[r]

        if moved:
            self.score += new_score
            if self.score > self.high_score:
                self.high_score = self.score
            self.spawn_tile()
            return True
        else:
            return False
    
    def get_state(self):
        return [cell for row in self.grid for cell in row] + [self.score]

    def is_game_over(self):
        # Check for empty cells
        if any(0 in row for row in self.grid):
            return False
            
        # Check for possible merges
        for r in range(GRID_SIZE):
            for c in range(GRID_SIZE):
                current = self.grid[r][c]
                # Check right neighbor
                if c < GRID_SIZE-1 and current == self.grid[r][c+1]:
                    return False
                # Check bottom neighbor
                if r < GRID_SIZE-1 and current == self.grid[r+1][c]:
                    return False
        return True
