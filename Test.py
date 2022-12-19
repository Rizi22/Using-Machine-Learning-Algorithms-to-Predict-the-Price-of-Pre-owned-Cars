import sys
from PyQt5.QtWidgets import QApplication, QWidget
from PyQt5 import uic

class Test(QWidget):
    def __init__(self):
        super().__init__()
        uic.loadUi('UITest.ui', self)
        
        self.button.clicked.connect(self.printValue)
    
    def printValue(self):
        print(self.lineEdit.text())

if __name__ == '__main__':
    test = QApplication(sys.argv)
    
    demo = Test()
    demo.show()
    
    try:
        sys.exit(test.exec_())
    except SystemExit:
        print("Closinggg")