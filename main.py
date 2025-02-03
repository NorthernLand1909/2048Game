from PyQt5.QtWidgets import QApplication
import sys
from core.game import Game2048
from ai.AIController import *

if __name__ == "__main__":
    # 人类玩家模式
    # app = QApplication(sys.argv)
    # game = Game2048()
    # game.show()
    # sys.exit(app.exec_())

    # AI模式
    # 初始化AI
    
    # 启动游戏
    app = QApplication(sys.argv)
    game = Game2048(model="transformer")
    game.show()
    sys.exit(app.exec_())