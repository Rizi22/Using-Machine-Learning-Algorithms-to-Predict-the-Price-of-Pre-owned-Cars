import sys
from PyQt5.QtWidgets import QApplication, QWidget
from PyQt5 import uic

# import NN
from NN import NN
NN = NN()

# NNO = NN()

class runUI(QWidget):
    

    def __init__(self):
        super().__init__()
        uic.loadUi('nearestNeighbourUI.ui', self)
        
        self.button.clicked.connect(self.runIt)
    
    def runIt(self):
        print(self.brandBox.text())
        NN.run(self.brandBox.text(), self.modelBox.text(), self.yearBox.text(), self.transmissionBox.text(), self.mileageBox.text(), self.fuelBox.text(), self.taxBox.text(), self.mpgBox.text(), self.engineBox.text())
        
        # self.brandBox.text()
        # self.modelBox.text()
        # self.yearBox.text()
        # self.transmissionBox.text()
        # self.mileageBox.text()
        # self.fuelBox.text()
        # self.taxBox.text()
        # self.mpgBox.text()
        # self.engineBox.text()

        # inputPred = []
        # entries = []

        # test= self.brandBox.text()
        # # test2 = NN.userInput(test)
        # # X_train, X_test, Y_train, Y_test = NN.dataset(test2)

        # print("\n List of models:")
        # print(list(NN.modelEncoder.classes_))

        # inputPred.append((NN.modelEncoder.transform(self.modelBox.text()))[0])
        # inputPred.append(int(self.yearBox.text()))
        # inputPred.append((NN.transmissionEncoder.transform(self.transmissionBox.text()))[0])
        # inputPred.append(int(self.mileageBox.text()))
        # inputPred.append((NN.fuelTypeEncoder.transform(self.fuelBox.text()))[0])
        # inputPred.append(int(self.taxBox.text()))
        # inputPred.append(float(self.mpgBox.text()))
        # inputPred.append(float(self.engineBox.text()))
        # entries.append(inputPred)
        # inputPred = NN.scaler.transform([inputPred])

        # import time
        # print("\n ***Predicting***")
        # start = time.time()
        # y_pred = NN.predict(X_train, inputPred, Y_train, 4)
        # # {0:.2f}'.format()
        # print("\n Predicted price for your car is: £", y_pred[0])

        # print("\n ***Predicted in", time.time() - start,"seconds***")
    

if __name__ == '__main__':
    test = QApplication(sys.argv)
    
    demo = runUI()
    demo.show()
    
    try:
        sys.exit(test.exec_())
    except SystemExit:
        print("Closinggg")