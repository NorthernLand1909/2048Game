from PyQt5.QtWidgets import QApplication
import sys

from AIController import AIController
from Game2048 import Game2048



if __name__ == "__main__":
    # 人类玩家模式
    app = QApplication(sys.argv)
    game = Game2048()
    game.show()
    sys.exit(app.exec_())

    # AI模式
    # app = QApplication(sys.argv)
    # ai = AIController()
    # game = Game2048(ai_controller=ai)
    # game.show()
    # sys.exit(app.exec_())