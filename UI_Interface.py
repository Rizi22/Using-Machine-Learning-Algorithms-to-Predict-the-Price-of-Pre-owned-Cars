import sys
from PyQt5.QtWidgets import QApplication, QWidget, QDialog
from PyQt5 import uic
from PyQt5 import QtWidgets
from NN import NN
NN = NN()

class mainMenuUI(QDialog):
    
    def __init__(self):
        super().__init__()
        uic.loadUi('mainMenuUI.ui', self)

        self.setWindowTitle('Pre-Owned Car Price Predictor')

        self.pushButton.clicked.connect(self.goToPage)
    
    def goToPage(self):
        if (self.KNNButton.isChecked()):
                widget.setCurrentIndex(widget.currentIndex()+1)

class NN_UI(QDialog):

    def __init__(self):
        super().__init__()
        uic.loadUi('nearestNeighbourUI.ui', self)

        self.setWindowTitle('KNN Pre-Owned Car Price Predictor')

        self.brandCombo.addItems(["Audi", "BMW", "Ford", "Hyundai", "Mercedes", "Skoda", "Toyota", "Vauxhall", "Volkswagen"])
        self.transmissionCombo.addItems(["Manual", "Automatic", "Semi-Auto"])
        self.fuelCombo.addItems(["Petrol", "Diesel", "Hybrid"])
        
        self.test()
        self.brandCombo.currentIndexChanged.connect(self.test)

        # self.transmissionCombo.setCurrentIndex (-1)

        # self.fuelCombo.setCurrentIndex (-1)

        self.backButton.clicked.connect(self.backPage)

        self.button.clicked.connect(self.runUI)
    
    def runUI(self):

        prediction = NN.UIInput(self.brandCombo.currentText(), self.modelCombo.currentText(), self.yearBox.text(), 
               self.transmissionCombo.currentText(), self.mileageBox.text(), self.fuelCombo.currentText(), 
               self.taxBox.text(), self.mpgBox.text(), self.engineBox.text())
        
        pred.label_2.setText("£" + str(prediction))
        widget.setCurrentIndex(widget.currentIndex()+1)

    def test(self):
        NN.testing(self.brandCombo.currentText())
        self.modelCombo.clear()
        self.modelCombo.addItems(list(NN.modelEncoder.classes_))
        print("test")
    
    def backPage(self):
        widget.setCurrentIndex(widget.currentIndex() - 1)

class predPage(QDialog):

    def __init__(self):
        super().__init__()
        uic.loadUi('predictionUI.ui', self)

        self.pushButton.clicked.connect(self.backPage)
        self.startButton.clicked.connect(self.homePage)
    
    def backPage(self):
        widget.setCurrentIndex(widget.currentIndex() - 1)
    
    def homePage(self):
        (widget.currentIndex() - 1).update()
        widget.setCurrentIndex(widget.currentIndex() - 2)

        
if __name__ == '__main__':

    app = QApplication(sys.argv)
    widget = QtWidgets.QStackedWidget()

    UI = mainMenuUI()
    NNUI = NN_UI()
    pred = predPage()

    widget.addWidget(UI)
    widget.addWidget(NNUI)
    widget.addWidget(pred)
    widget.setFixedHeight(700)
    widget.setFixedWidth(1100)
    widget.show()

    try:
        sys.exit(app.exec_())
    except SystemExit:
        print("Closinggg")

