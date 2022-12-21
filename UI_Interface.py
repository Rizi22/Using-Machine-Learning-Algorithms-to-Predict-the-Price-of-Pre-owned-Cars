import sys
from PyQt5.QtWidgets import QApplication, QWidget, QDialog
from PyQt5 import uic, QtCore
from PyQt5 import QtWidgets
from NN import NN
NN = NN()
from DT import DT
DT = DT()

class mainMenuUI(QDialog):
    
    def __init__(self):
        super().__init__()
        uic.loadUi('mainMenuUI.ui', self)
        
        self.setWindowTitle('Pre-Owned Car Price Predictor')
        self.label.setAlignment(QtCore.Qt.AlignCenter)

        self.pushButton.clicked.connect(self.goToPage)
    
    def goToPage(self):
        if (self.KNNButton.isChecked()):
                Input_UI.label1.setText("KNN Car Price Predictor")
                Input_UI.setWindowTitle('KNN Pre-Owned Car Price Predictor')
                Input_UI.algorithm = "KNN"
                widget.setCurrentIndex(1)

        elif (self.DTButton.isChecked()):
                Input_UI.label1.setText("Decision Tree Car Price Predictor")
                Input_UI.setWindowTitle('DT Pre-Owned Car Price Predictor')
                Input_UI.algorithm = "DT"
                widget.setCurrentIndex(1)
            

class InputUI(QDialog):

    def __init__(self):
        super().__init__()
        uic.loadUi('InputUI.ui', self)
        self.algorithm = ''
        self.label1.setAlignment(QtCore.Qt.AlignCenter)

        self.brandCombo.addItems(["Audi", "BMW", "Ford", "Hyundai", "Mercedes", "Skoda", "Toyota", "Vauxhall", "Volkswagen"])
        self.transmissionCombo.addItems(["Manual", "Automatic", "Semi-Auto"])
        self.fuelCombo.addItems(["Petrol", "Diesel", "Hybrid"])
        
        self.test()
        self.brandCombo.currentIndexChanged.connect(self.test)
        self.backButton.clicked.connect(self.backPage)
        self.button.clicked.connect(self.runUI)
    
    def runUI(self):
        prediction = 0
        print (self.algorithm)
        if self.algorithm == "KNN":
            prediction = NN.UIInput(self.brandCombo.currentText(), self.modelCombo.currentText(), self.yearBox.text(), 
               self.transmissionCombo.currentText(), self.mileageBox.text(), self.fuelCombo.currentText(), 
               self.taxBox.text(), self.mpgBox.text(), self.engineBox.text())
        elif self.algorithm == "DT":
            prediction = DT.UIInput(self.brandCombo.currentText(), self.modelCombo.currentText(), self.yearBox.text(), 
               self.transmissionCombo.currentText(), self.mileageBox.text(), self.fuelCombo.currentText(), 
               self.taxBox.text(), self.mpgBox.text(), self.engineBox.text())
        
        pred.label_2.setText("£" + str(prediction))
        widget.setCurrentIndex(2)

    def test(self):
        DT.testing(self.brandCombo.currentText())
        self.modelCombo.clear()
        self.modelCombo.addItems(list(DT.modelEncoder.classes_))
    
    def backPage(self):
        widget.setCurrentIndex(0)

class predPage(QDialog):

    def __init__(self):
        super().__init__()
        uic.loadUi('predictionUI.ui', self)
        self.label.setAlignment(QtCore.Qt.AlignCenter)
        self.label_2.setAlignment(QtCore.Qt.AlignCenter)

        self.pushButton.clicked.connect(self.backPage)
        self.startButton.clicked.connect(self.homePage)
    
    def backPage(self):
        widget.setCurrentIndex(widget.currentIndex() - 1)
    
    def homePage(self):
        widget.update()
        widget.setCurrentIndex(0)

        
if __name__ == '__main__':

    app = QApplication(sys.argv)
    widget = QtWidgets.QStackedWidget()

    UI = mainMenuUI()
    Input_UI = InputUI()
    pred = predPage()

    widget.addWidget(UI)
    widget.addWidget(Input_UI)
    widget.addWidget(pred)
    widget.setFixedHeight(700)
    widget.setFixedWidth(1100)
    widget.show()

    try:
        sys.exit(app.exec_())
    except SystemExit:
        print("Closinggg")

