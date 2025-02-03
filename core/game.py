from utils import GRID_SIZE, TILE_COLORS
from core.board import Board
from ai.AIController import TrainedAgent
from PyQt5.QtWidgets import QWidget, QVBoxLayout, QLabel, QPushButton, QGridLayout
from PyQt5.QtCore import Qt, QTimer

class Game2048(QWidget):
    def __init__(self, model=None):
        super().__init__()
        self.board = Board()
        self.model = model
        self.init_ai_controller()
        self.ai_timer = None
        self.restart_timer = QTimer()
        self.init_ui()
        self.update_board()

    def init_ai_controller(self):
        if self.model:
            self.ai_controller = TrainedAgent(self.model)
        else:
            self.ai_controller = None

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
            self.start_ai()

    def start_ai(self):
        """启动AI游戏循环"""
        self.ai_timer = self.startTimer(10)  # AI移动间隔
        self.restart_timer.timeout.connect(self.handle_restart)

    def timerEvent(self, event):
        """处理AI移动"""
        if self.board.is_game_over():
            self.handle_game_over()
            return
        
        # 获取原始网格和归一化网格
        raw_grid = self.board.grid.copy()
        norm_state = self.board.get_normalized_state(self.model)
        
        # 获取AI决策
        direction = self.ai_controller.get_move(raw_grid, norm_state)
        if direction:
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

    def handle_game_over(self):
        """处理游戏结束"""
        # 停止当前AI计时器
        if self.ai_timer:
            self.killTimer(self.ai_timer)
            self.ai_timer = None
        
        # 3秒后自动重启
        self.restart_timer.start(3000)  # 3秒后触发重启

    def handle_restart(self):
        """重启游戏"""
        self.restart_timer.stop()
        # 保留最高分
        current_high = self.board.high_score
        # 创建新游戏
        self.board = Board(load_saved=False)
        self.board.high_score = current_high
        self.update_board()
        # 重新启动AI
        if self.ai_controller:
            self.start_ai()