import sys
import os
import numpy as np
import pandas as pd
import time

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import MinMaxScaler

from PyQt5.QtWidgets import QApplication, QWidget, QDialog
from PyQt5 import uic, QtCore
from PyQt5 import QtWidgets

from nearestNeighbour import nearestNeighbour
from decisionTree import decisionTree
from randomForest import randomForest
from linearRegression import linearRegression

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
        # Input_UI.pred.clear()
        # Input_UI.pred2.clear()
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
        
        elif (self.LRButton.isChecked()):
                Input_UI.label1.setText("Linear Regression Car Price Predictor")
                Input_UI.setWindowTitle('LR Pre-Owned Car Price Predictor')
                Input_UI.algorithm = "LR"
                widget.setCurrentIndex(1)
        
        elif (self.combinedButton.isChecked()):
                Input_UI.label1.setText("Combined ML Car Price Predictor")
                Input_UI.setWindowTitle('Combined ML Pre-Owned Car Price Predictor')
                Input_UI.algorithm = "combined"
                widget.setCurrentIndex(1)
            

class InputUI(QDialog):

    def __init__(self):
        super().__init__()
        uic.loadUi(path('UI_Files/InputUI.ui'), self)
        self.algorithm = ''
        self.label1.setAlignment(QtCore.Qt.AlignCenter)
        self.pred.setAlignment(QtCore.Qt.AlignCenter)
        self.pred2.setAlignment(QtCore.Qt.AlignCenter)

        self.modelEncoder = LabelEncoder()
        self.transmissionEncoder = LabelEncoder()
        self.fuelTypeEncoder = LabelEncoder()
        self.scaler = MinMaxScaler()

        self.brandCombo.addItems(["Audi", "BMW", "Ford", "Hyundai", "Mercedes", "Skoda", "Toyota", "Vauxhall", "Volkswagen"])
        self.transmissionCombo.addItems(["Manual", "Automatic", "Semi-Auto"])
        self.fuelCombo.addItems(["Petrol", "Diesel", "Hybrid"])
        
        self.addModel()
        self.brandCombo.currentIndexChanged.connect(self.addModel)
        self.backButton.clicked.connect(self.backPage)
        self.button.clicked.connect(self.runUI)


    def runUI(self):

        self.pred.setText("Predicting...")
        self.pred2.setText("Please wait...")
        QApplication.processEvents()

        inputPred = []

        print(self.chosenBrand(self.brandCombo.currentText()))
        X_train, X_test, Y_train, Y_test = self.loadDataset(self.chosenBrand(self.brandCombo.currentText()))   
        
        inputPred.append((self.modelEncoder.transform([self.modelCombo.currentText()]))[0])
        inputPred.append(int(self.yearBox.text()))
        inputPred.append((self.transmissionEncoder.transform([self.transmissionCombo.currentText()]))[0])
        inputPred.append(int(self.mileageBox.text()))
        inputPred.append((self.fuelTypeEncoder.transform([self.fuelCombo.currentText()]))[0])
        inputPred.append(int(self.taxBox.text()))
        inputPred.append(float(self.mpgBox.text()))
        inputPred.append(float(self.engineBox.text()))
        
        print("\n ***Predicting***")
        
        timer = time.time()

        match self.algorithm:   
            case "LR":
                LR = linearRegression()
                LR.fit(X_train, Y_train)
                inputPred = self.scaler.transform([inputPred])
                Y_pred = LR.predict(inputPred)
            case "KNN":
                KNN = nearestNeighbour()
                inputPred = self.scaler.transform([inputPred])
                Y_pred = KNN.predict(X_train, inputPred, Y_train, 4)
            case "DT":
                DT = decisionTree(3, 93)
                DT.fit(X_train, Y_train)
                Y_pred = DT.predict([inputPred])
            case "RF":
                RF = randomForest()
                RF.fit(X_train, Y_train)
                Y_pred = RF.predict([inputPred])
            
            case "combined":
                X_train, X_test, Y_train, Y_test, scaled_X_train, scaled_X_test, scaled_Y_train, scaled_Y_test = self.loadDataset(self.chosenBrand(self.brandCombo.currentText())) 
                
                print("GOT HEREE 1")
                DT = decisionTree(3, 93)
                DT.fit(X_train, Y_train)
                Y_pred1 = DT.predict([inputPred])

                print("GOT HEREE 2")
                RF = randomForest()
                RF.fit(X_train, Y_train)
                Y_pred2 = RF.predict([inputPred])

                inputPred = self.scaler.transform([inputPred])

                LR = linearRegression()
                LR.fit(scaled_X_train, scaled_Y_train)
                Y_pred3 = LR.predict(inputPred)
                
                KNN = nearestNeighbour()
                Y_pred4 = KNN.predict(scaled_X_train, inputPred, scaled_Y_train, 4)
                
                print(Y_pred1, Y_pred2, Y_pred3, Y_pred4)
                Y_pred = (Y_pred1 + Y_pred2 + Y_pred3 + Y_pred4) / 4
                


        print("\n Predicted price for your car is: £", round(Y_pred[0], 2))
        print("\n ***Predicted in", time.time() - timer,"seconds***")

        pred.label_2.setText("£" + str(round(Y_pred[0], 2)))
        widget.setCurrentIndex(2)


    def chosenBrand(self, brand):

        match brand:
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
                return "UKUsedCarDataSet/vw.csv"
            case _:
                print("Invalid input")
                return
    
    def loadDataset(self, brand):

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
            case "DT" | "RF":
                X = file.drop(['price'], axis = 1).to_numpy()
                Y = file['price'].values.reshape(-1,1)
                X_train, X_test, Y_train, Y_test = train_test_split(X, Y, random_state = 601)
                return  X_train, X_test, Y_train, Y_test
            case "KNN" | "LR":
                X = file.drop(columns = ['price'])
                Y = file.price
                X_train, X_test, Y_train, Y_test = train_test_split(X, Y, random_state = 601)
                self.scaler.fit(X_train)
                X_train = self.scaler.transform(X_train)
                X_test = self.scaler.transform(X_test)
                return  X_train, X_test, Y_train, Y_test
            
            case "combined":
                X = file.drop(['price'], axis = 1).to_numpy()
                Y = file['price'].values.reshape(-1,1)
                X_train, X_test, Y_train, Y_test = train_test_split(X, Y, random_state = 601)

                X = file.drop(columns = ['price'])
                Y = file.price
                scaled_X_train, scaled_X_test, scaled_Y_train, scaled_Y_test = train_test_split(X, Y, random_state = 601)
                self.scaler.fit(scaled_X_train)
                scaled_X_train = self.scaler.transform(scaled_X_train)
                scaled_X_test = self.scaler.transform(scaled_X_test)
                
                return  X_train, X_test, Y_train, Y_test, scaled_X_train, scaled_X_test, scaled_Y_train, scaled_Y_test
        

    def addModel(self):
        self.loadDataset(self.chosenBrand(self.brandCombo.currentText()))
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

        self.pushButton.clicked.connect(self.clearText)
        self.pushButton.clicked.connect(self.backPage)

        self.startButton.clicked.connect(self.refreshPage)
        self.startButton.clicked.connect(self.homePage)


    def refreshPage(self):
        Input_UI.pred.clear()
        Input_UI.pred2.clear()
        Input_UI.yearBox.clear()
        Input_UI.mileageBox.clear()
        Input_UI.taxBox.clear()
        Input_UI.mpgBox.clear()
        Input_UI.engineBox.clear()

    def clearText(self):
        Input_UI.pred.clear()
        Input_UI.pred2.clear()
    
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
        print("***Program Terminated***")