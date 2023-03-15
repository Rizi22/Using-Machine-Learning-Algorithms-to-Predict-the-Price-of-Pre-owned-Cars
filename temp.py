def runUI(self):
    prediction = 0
    print (self.algorithm)

    brand = self.brandCombo.currentText()
    model = self.modelCombo.currentText()
    year = self.yearBox.text()
    transmission = self.transmissionCombo.currentText()
    mileage = self.mileageBox.text()
    fuel = self.fuelCombo.currentText()
    tax = self.taxBox.text()
    mpg = self.mpgBox.text()
    engine = self.engineBox.text()


    inputPred = []
    X_train, X_test, Y_train, Y_test = self.dataset(self.userInput(chooseBrand))
        
    print("\n ***Training Tree Model***")
    timer = time.time()

    match self.algorithm:   
        case "DT":
            decisionTree.fit(X_train, Y_train)
        case "RF":
            randomForest.fit(X_train, Y_train)
    
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
            Y_pred = KNN.predict((X_train, inputPred, Y_train, 4)
        case "DT":
            decisionTree.fit(X_train, Y_train)
            Y_pred = decisionTree.predict([inputPred])
        case "RF":
            randomForest.fit(X_train, Y_train)
            Y_pred = randomForest.predict([inputPred])

    print("\n Predicted price for your car is: £",round(Y_pred[0], 2))
    print("\n ***Predicted in", time.time() - timer,"seconds***")

    pred.label_2.setText("£" + str(Y_pred[0]))
    widget.setCurrentIndex(2)



    if self.algorithm == "KNN":
        prediction = NN.UIInput(self.brandCombo.currentText(), self.modelCombo.currentText(), self.yearBox.text(), 
            self.transmissionCombo.currentText(), self.mileageBox.text(), self.fuelCombo.currentText(), 
            self.taxBox.text(), self.mpgBox.text(), self.engineBox.text())
    elif self.algorithm == "DT":
        prediction = decisionTree.UIInput(self.brandCombo.currentText(), self.modelCombo.currentText(), self.yearBox.text(), 
            self.transmissionCombo.currentText(), self.mileageBox.text(), self.fuelCombo.currentText(), 
            self.taxBox.text(), self.mpgBox.text(), self.engineBox.text())
    elif self.algorithm == "RF":
        prediction = randomForest.UIInput(self.brandCombo.currentText(), self.modelCombo.currentText(), self.yearBox.text(), 
            self.transmissionCombo.currentText(), self.mileageBox.text(), self.fuelCombo.currentText(), 
            self.taxBox.text(), self.mpgBox.text(), self.engineBox.text())
    
    pred.label_2.setText("£" + str(prediction))
    widget.setCurrentIndex(2)