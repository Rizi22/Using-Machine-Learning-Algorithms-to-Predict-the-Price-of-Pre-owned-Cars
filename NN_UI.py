import sys
from PyQt5.QtWidgets import QApplication, QWidget
from PyQt5 import uic
from NN import NN
NN = NN()

class NN_UI(QWidget):
    

    def __init__(self):
        super().__init__()
        uic.loadUi('nearestNeighbourUI.ui', self)
        
        self.button.clicked.connect(self.runUI)
    
    def runUI(self):
        print(self.brandBox.text())
        NN.UIInput(self.brandBox.text(), self.modelBox.text(), self.yearBox.text(), 
               self.transmissionBox.text(), self.mileageBox.text(), self.fuelBox.text(), 
               self.taxBox.text(), self.mpgBox.text(), self.engineBox.text())     

if __name__ == '__main__':
    test = QApplication(sys.argv)
    
    UI = NN_UI()
    UI.show()
    
    try:
        sys.exit(test.exec_())
    except SystemExit:
        print("Closinggg")