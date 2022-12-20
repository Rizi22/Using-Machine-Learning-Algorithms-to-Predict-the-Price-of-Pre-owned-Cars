import sys
from PyQt5.QtWidgets import QApplication, QWidget, QDialog
from PyQt5 import uic

class mainMenuUI(QDialog):
    
    def __init__(self):
        super().__init__()
        uic.loadUi('mainMenuUI.ui', self)

        self.setWindowTitle('Pre-Owned Car Price Predictor')

        self.pushButton.clicked.connect(self.goToPage)
    
    def goToPage(self):
        print("test")

if __name__ == '__main__':
    test = QApplication(sys.argv)
    
    UI = mainMenuUI()
    UI.show()
    
    try:
        sys.exit(test.exec_())
    except SystemExit:
        print("Closinggg")