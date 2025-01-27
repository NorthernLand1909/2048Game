from PyQt5.QtWidgets import QApplication, QWidget, QVBoxLayout, QLabel, QPushButton, QGridLayout
from PyQt5.QtCore import QPropertyAnimation, QRect, Qt, QParallelAnimationGroup
import sys
import random
import json
import os

GRID_SIZE = 4
NEW_TILE = 2
SAVE_FILE = "2048_save.json"
TILE_COLORS = {
    0: "#CDC1B4", 2: "#EEE4DA", 4: "#EDE0C8", 8: "#F2B179",
    16: "#F59563", 32: "#F67C5F", 64: "#F65E3B", 128: "#EDCF72",
    256: "#EDCC61", 512: "#EDC850", 1024: "#EDC53F", 2048: "#EDC22E"
}

class Board:
    def __init__(self):
        self.grid = [[0] * GRID_SIZE for _ in range(GRID_SIZE)]
        self.previous_grid = None

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

    def save_game(self, score, high_score):
        with open(SAVE_FILE, "w") as file:
            json.dump({"grid": self.grid, "score": score, "high_score": high_score}, file)

    def load_game(self):
        if os.path.exists(SAVE_FILE):
            try:
                with open(SAVE_FILE, "r") as file:
                    data = json.load(file)
                    loaded_grid = data.get("grid", [])
                    if len(loaded_grid) == GRID_SIZE and all(len(row) == GRID_SIZE for row in loaded_grid):
                        self.grid = loaded_grid
                        return data.get("score", 0), data.get("high_score", 0)
            except (json.JSONDecodeError, KeyError):
                pass
        return 0, 0

class Game2048(QWidget):
    def __init__(self):
        super().__init__()
        self.board = Board()
        self.score, self.high_score = self.board.load_game()
        self.init_ui()
        self.update_board()

    def init_ui(self):
        self.setWindowTitle("2048 Game")
        self.layout = QVBoxLayout()

        self.score_label = QLabel(f"Score: {self.score}")
        self.layout.addWidget(self.score_label)

        self.high_score_label = QLabel(f"High Score: {self.high_score}")
        self.layout.addWidget(self.high_score_label)

        self.grid_layout = QGridLayout()
        self.tiles = []
        for i in range(GRID_SIZE):
            row = []
            for j in range(GRID_SIZE):
                label = QLabel("", self)
                label.setStyleSheet(f"background-color: {TILE_COLORS[0]}; font-size: 24px; padding: 20px;")
                label.setAlignment(Qt.AlignCenter)
                self.grid_layout.addWidget(label, i, j)
                row.append(label)
            self.tiles.append(row)
        self.layout.addLayout(self.grid_layout)

        self.undo_button = QPushButton("Undo (Z)")
        self.undo_button.clicked.connect(self.undo_move)
        self.layout.addWidget(self.undo_button)

        self.restart_button = QPushButton("Restart (R)")
        self.restart_button.clicked.connect(self.restart_game)
        self.layout.addWidget(self.restart_button)

        self.setLayout(self.layout)
        self.setFocusPolicy(Qt.StrongFocus)

    def keyPressEvent(self, event):
        key_map = {
            Qt.Key_Left: "Left", Qt.Key_Right: "Right", Qt.Key_Up: "Up", Qt.Key_Down: "Down",
            Qt.Key_A: "Left", Qt.Key_D: "Right", Qt.Key_W: "Up", Qt.Key_S: "Down",
            Qt.Key_Z: "Undo", Qt.Key_R: "Restart"
        }
        if event.key() in key_map:
            action = key_map[event.key()]
            if action in ["Left", "Right", "Up", "Down"]:
                self.move(action)
            elif action == "Undo":
                self.undo_move()
            elif action == "Restart":
                self.restart_game()

    def update_board(self):
        for r in range(GRID_SIZE):
            for c in range(GRID_SIZE):
                value = self.board.grid[r][c]
                self.tiles[r][c].setText(str(value) if value else "")
                self.tiles[r][c].setStyleSheet(f"background-color: {TILE_COLORS.get(value, '#3C3A32')}; font-size: 24px; padding: 20px;")
        self.score_label.setText(f"Score: {self.score}")
        self.high_score_label.setText(f"High Score: {self.high_score}")
        self.board.save_game(self.score, self.high_score)

    def move(self, direction):
        self.board.save_state()
        moved = False

        def compress(row):
            new_row = [v for v in row if v != 0]
            new_row += [0] * (GRID_SIZE - len(new_row))
            return new_row

        def merge(row):
            new_row = compress(row)
            for i in range(GRID_SIZE - 1):
                if new_row[i] and new_row[i] == new_row[i + 1]:
                    new_row[i] *= 2
                    self.score += new_row[i]
                    new_row[i + 1] = 0
            return compress(new_row)

        for i in range(GRID_SIZE):
            if direction in ('Left', 'Right'):
                row = self.board.grid[i][:]
                if direction == 'Right':
                    row.reverse()
                row = merge(row)
                if direction == 'Right':
                    row.reverse()
                if row != self.board.grid[i]:
                    moved = True
                self.board.grid[i] = row
            else:
                col = [self.board.grid[r][i] for r in range(GRID_SIZE)]
                if direction == 'Down':
                    col.reverse()
                col = merge(col)
                if direction == 'Down':
                    col.reverse()
                if col != [self.board.grid[r][i] for r in range(GRID_SIZE)]:
                    moved = True
                for r in range(GRID_SIZE):
                    self.board.grid[r][i] = col[r]

        if moved:
            self.board.spawn_tile()
            if self.score > self.high_score:
                self.high_score = self.score
            self.update_board()

    def undo_move(self):
        self.board.undo()
        self.update_board()

    def restart_game(self):
        self.board = Board()
        self.score = 0
        self.board.spawn_tile()
        self.board.spawn_tile()
        self.update_board()

if __name__ == "__main__":
    app = QApplication(sys.argv)
    game = Game2048()
    game.show()
    sys.exit(app.exec_())
