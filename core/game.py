from utils import GRID_SIZE, TILE_COLORS
from core.board import Board
from PyQt5.QtWidgets import QWidget, QVBoxLayout, QLabel, QPushButton, QGridLayout
from PyQt5.QtCore import Qt

class Game2048(QWidget):
    def __init__(self, ai_controller=None):
        super().__init__()
        self.board = Board()
        self.ai_controller = ai_controller
        self.init_ui()
        self.update_board()

    def init_ui(self):
        self.setWindowTitle("2048 Game")
        self.layout = QVBoxLayout()

        self.score_label = QLabel(f"Score: {self.board.score}")
        self.layout.addWidget(self.score_label)

        self.high_score_label = QLabel(f"High Score: {self.board.high_score}")
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

        # AI control
        if self.ai_controller:
            self.ai_timer = self.startTimer(100)  # AI moves every 100ms

    def timerEvent(self, event):
        if self.ai_controller:
            state = self.board.get_state()
            direction = self.ai_controller.get_move(state)
            self.move(direction)

    def keyPressEvent(self, event):
        key_map = {
            Qt.Key_Left: "Left", Qt.Key_Right: "Right", 
            Qt.Key_Up: "Up", Qt.Key_Down: "Down",
            Qt.Key_A: "Left", Qt.Key_D: "Right",
            Qt.Key_W: "Up", Qt.Key_S: "Down",
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
        self.score_label.setText(f"Score: {self.board.score}")
        self.high_score_label.setText(f"High Score: {self.board.high_score}")
        self.board.save_game()

    def move(self, direction):
        moved = self.board.move(direction)
        if moved:
            self.update_board()
        return moved

    def undo_move(self):
        self.board.undo()
        self.update_board()

    def restart_game(self):
        # 保存当前最高分
        current_high_score = self.board.high_score
        # 创建新棋盘时不加载存档
        self.board = Board(load_saved=False)
        # 恢复最高分
        self.board.high_score = current_high_score
        # 更新界面
        self.update_board()
