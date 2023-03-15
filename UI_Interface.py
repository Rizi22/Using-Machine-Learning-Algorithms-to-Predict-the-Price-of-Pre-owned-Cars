import sys
import os

from sklearn.preprocessing import LabelEncoder
import time
from sklearn.model_selection import train_test_split
import numpy as np
import pandas as pd

from PyQt5.QtWidgets import QApplication, QWidget, QDialog
from PyQt5 import uic, QtCore
from PyQt5 import QtWidgets
from NN import NN
NN = NN()

from decisionTree import DTRegressor


from randomForest import randomForest
randomForest = randomForest()

def path(relative_path):
    try:
        # PyInstaller creates a temp folder and stores path in _MEIPASS
        base_path = sys._MEIPASS
    except Exception:
        base_path = os.path.abspath(".")
    return os.path.join(base_path, relative_path)

class mainMenuUI(QDialog):
    
    def __init__(self):
        super().__init__()
        uic.loadUi(path('UI_Files/mainMenuUI.ui'), self)
        
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
        
        elif (self.RFButton.isChecked()):
                Input_UI.label1.setText("Random Forest Car Price Predictor")
                Input_UI.setWindowTitle('RF Pre-Owned Car Price Predictor')
                Input_UI.algorithm = "RF"
                widget.setCurrentIndex(1)
            

class InputUI(QDialog):

    def __init__(self):
        super().__init__()
        uic.loadUi(path('UI_Files/InputUI.ui'), self)
        self.algorithm = ''
        self.label1.setAlignment(QtCore.Qt.AlignCenter)
        self.modelEncoder = LabelEncoder()
        self.transmissionEncoder = LabelEncoder()
        self.fuelTypeEncoder = LabelEncoder()

        self.brandCombo.addItems(["Audi", "BMW", "Ford", "Hyundai", "Mercedes", "Skoda", "Toyota", "Vauxhall", "Volkswagen"])
        self.transmissionCombo.addItems(["Manual", "Automatic", "Semi-Auto"])
        self.fuelCombo.addItems(["Petrol", "Diesel", "Hybrid"])
        
        self.addModel()
        self.brandCombo.currentIndexChanged.connect(self.addModel)
        self.backButton.clicked.connect(self.backPage)
        self.button.clicked.connect(self.runUI)
    
    # def runUI(self):
    #     prediction = 0
    #     print (self.algorithm)
    #     if self.algorithm == "KNN":
    #         prediction = NN.UIInput(self.brandCombo.currentText(), self.modelCombo.currentText(), self.yearBox.text(), 
    #            self.transmissionCombo.currentText(), self.mileageBox.text(), self.fuelCombo.currentText(), 
    #            self.taxBox.text(), self.mpgBox.text(), self.engineBox.text())
    #     elif self.algorithm == "DT":
    #         prediction = decisionTree.UIInput(self.brandCombo.currentText(), self.modelCombo.currentText(), self.yearBox.text(), 
    #            self.transmissionCombo.currentText(), self.mileageBox.text(), self.fuelCombo.currentText(), 
    #            self.taxBox.text(), self.mpgBox.text(), self.engineBox.text())
    #     elif self.algorithm == "RF":
    #         prediction = randomForest.UIInput(self.brandCombo.currentText(), self.modelCombo.currentText(), self.yearBox.text(), 
    #            self.transmissionCombo.currentText(), self.mileageBox.text(), self.fuelCombo.currentText(), 
    #            self.taxBox.text(), self.mpgBox.text(), self.engineBox.text())
        
    #     pred.label_2.setText("£" + str(prediction))
    #     widget.setCurrentIndex(2)


    def runUI(self):

        brand = self.brandCombo.currentText()
        model = self.modelCombo.currentText()
        year = self.yearBox.text()
        transmission = self.transmissionCombo.currentText()
        mileage = self.mileageBox.text()
        fuelType = self.fuelCombo.currentText()
        tax = self.taxBox.text()
        mpg = self.mpgBox.text()
        engineSize = self.engineBox.text()

        inputPred = []

        X_train, X_test, Y_train, Y_test = self.dataset(self.userInput(brand))
            
        print("\n ***Training Tree Model***")
        timer = time.time()
        
        inputPred.append((self.modelEncoder.transform([model]))[0])
        inputPred.append(int(year))
        inputPred.append((self.transmissionEncoder.transform([transmission]))[0])
        inputPred.append(int(mileage))
        inputPred.append((self.fuelTypeEncoder.transform([fuelType]))[0])
        inputPred.append(int(tax))
        inputPred.append(float(mpg))
        inputPred.append(float(engineSize))


        print("\n ***Predicting***")

        match self.algorithm:   
            case "KNN":
                Y_pred = NN.predict(X_train, inputPred, Y_train, 4)
            case "DT":
                print("HEREEEE")
                DT = DTRegressor(3, 93)
                DT.fit(X_train, Y_train)
                Y_pred = DT.predict([inputPred])
            case "RF":
                randomForest.fit(X_train, Y_train)
                Y_pred = randomForest.predict([inputPred])

        print("\n Predicted price for your car is: £",round(Y_pred[0], 2))
        print("\n ***Predicted in", time.time() - timer,"seconds***")

        pred.label_2.setText("£" + str(Y_pred[0]))
        widget.setCurrentIndex(2)

    def testing(self, chooseBrand):
        self.dataset(self.userInput(chooseBrand))
        return

    def userInput(self, chooseBrand):

        match chooseBrand:
            case "Audi":
                return "UKUsedCarDataSet/audi.csv"
            case "BMW":
                return "UKUsedCarDataSet/bmw.csv"
            case "Ford":
                return "UKUsedCarDataSet/ford.csv"
            case "Hyundai":
                return "UKUsedCarDataSet/hyundi.csv"
            case "Mercedes":
                return "UKUsedCarDataSet/merc.csv"
            case "Skoda":
                return "UKUsedCarDataSet/skoda.csv"
            case "Toyota":
                return "UKUsedCarDataSet/toyota.csv"
            case "Vauxhall":
                return "UKUsedCarDataSet/vauxhall.csv"
            case "Volkswagen":
                return "UKUsedCarDataSet/volkswagen.csv"
            case _:
                print("Invalid input")
                return
    
    def dataset(self, brand):

        file = pd.read_csv(brand, quotechar='"', skipinitialspace=True)

        for i in ['year']:
            q75,q25 = np.percentile(file.loc[:,i],[75,25])
            IQR = q75-q25
        
            maxQ = q75+(1.5*IQR)
            minQ = q25-(1.5*IQR)
        
            file.loc[file[i] < minQ, i] = np.nan
            file.loc[file[i] > maxQ, i] = np.nan

        file = file.dropna(axis = 0)

        self.modelEncoder.fit(file["model"])
        file["model"] = self.modelEncoder.transform(file["model"])

        self.transmissionEncoder.fit(file["transmission"])
        file["transmission"] = self.transmissionEncoder.transform(file["transmission"])

        self.fuelTypeEncoder.fit(file["fuelType"])
        file["fuelType"] = self.fuelTypeEncoder.transform(file["fuelType"])

        file = file.head(5000)

        match self.algorithm:
            case "DT":
                X = file.drop(['price'], axis = 1).to_numpy()
                Y = file['price'].values.reshape(-1,1)
                X_train, X_test, Y_train, Y_test = train_test_split(X, Y, random_state = 601)
                return  X_train, X_test, Y_train, Y_test
            case "RF":
                X = file.drop(['price'], axis = 1).to_numpy()
                Y = file['price'].values.reshape(-1,1)
                X_train, X_test, Y_train, Y_test = train_test_split(X, Y, random_state = 601)
                return  X_train, X_test, Y_train, Y_test
            case "KNN":
                X = file.drop(columns = ['price'])
                Y = file.price
                X_train, X_test, Y_train, Y_test = train_test_split(X, Y, random_state = 601)
                self.scaler.fit(X_train)
                X_train = self.scaler.transform(X_train)
                X_test = self.scaler.transform(X_test)
                return  X_train, X_test, Y_train, Y_test
        

    def addModel(self):
        self.testing(self.brandCombo.currentText())
        self.modelCombo.clear()
        self.modelCombo.addItems(list(self.modelEncoder.classes_))
    
    def backPage(self):
        widget.setCurrentIndex(0)

class predPage(QDialog):

    def __init__(self):
        super().__init__()
        uic.loadUi(path('UI_Files/predictionUI.ui'), self)
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

    UIMenu = mainMenuUI()
    Input_UI = InputUI()
    pred = predPage()

    widget.addWidget(UIMenu)
    widget.addWidget(Input_UI)
    widget.addWidget(pred)
    widget.setFixedHeight(700)
    widget.setFixedWidth(1100)
    widget.show()

    try:
        sys.exit(app.exec_())
    except SystemExit:
        print("Closinggg")