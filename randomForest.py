import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import csv
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import LabelEncoder

from collections import Counter
from decisionTree import DTRegressor
# DTRegressor = DTRegressor()

class randomForest():
    def __init__(self):
        self.modelEncoder = LabelEncoder()
        self.transmissionEncoder = LabelEncoder()
        self.fuelTypeEncoder = LabelEncoder()
        self.scaler = MinMaxScaler()
    
    def dataset(self, brand):

        file = pd.read_csv(brand, quotechar='"', skipinitialspace=True)

        self.modelEncoder.fit(file["model"])
        file["model"] = self.modelEncoder.transform(file["model"])

        self.transmissionEncoder.fit(file["transmission"])
        file["transmission"] = self.transmissionEncoder.transform(file["transmission"])

        self.fuelTypeEncoder.fit(file["fuelType"])
        file["fuelType"] = self.fuelTypeEncoder.transform(file["fuelType"])

        file = file.head(1000)
        X = file.drop(['price'], axis = 1).to_numpy()
        Y = file['price'].values.reshape(-1,1)

        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, random_state = 601)
        
        return  X_train, X_test, Y_train, Y_test
    
    def testing(self, chooseBrand):
        self.dataset(self.userInput(chooseBrand))
        return

    def userInput(self, chooseBrand):
        
        if chooseBrand == "Audi":
            return "UKUsedCarDataSet/audi.csv"
        elif chooseBrand == "BMW":
            return "UKUsedCarDataSet/bmw.csv"
        elif chooseBrand == "Ford":
            return "UKUsedCarDataSet/ford.csv"
        elif chooseBrand == "Hyundai":
            return "UKUsedCarDataSet/hyundi.csv"
        elif chooseBrand == "Mercedes":
            return "UKUsedCarDataSet/merc.csv"
        elif chooseBrand == "Skoda":
            return "UKUsedCarDataSet/skoda.csv"
        elif chooseBrand == "Toyota":
            return "UKUsedCarDataSet/toyota.csv"
        elif chooseBrand == "Vauxhall":
            return "UKUsedCarDataSet/vauxhall.csv"
        elif chooseBrand == "Volkswagen":
            return "UKUsedCarDataSet/vw.csv"
        else:
            print("Invalid Car Brand")
            return
        
    def UIInput(self, chooseBrand, model, year, transmission, mileage, fuelType, tax, mpg, engineSize):

        inputPred = []
        entries = []

        X_train, X_test, Y_train, Y_test = self.dataset(self.userInput(chooseBrand))
        print("\n ***Training Tree Model***")
        myForest = forestRegression(3, 93)  
        myForest.fit(X_train, Y_train)

        inputPred.append((self.modelEncoder.transform([model]))[0])
        inputPred.append(int(year))
        inputPred.append((self.transmissionEncoder.transform([transmission]))[0])
        inputPred.append(int(mileage))
        inputPred.append((self.fuelTypeEncoder.transform([fuelType]))[0])
        inputPred.append(int(tax))
        inputPred.append(float(mpg))
        inputPred.append(float(engineSize))
        entries.append(inputPred)

        import time
        print("\n ***Predicting***")
        start = time.time()
        y_pred = myForest.predict([inputPred])
        # {0:.2f}'.format()
        print("\n Predicted price for your car is: £", y_pred[0])

        print("\n ***Predicted in", time.time() - start,"seconds***")

        # RS6,2016,Semi-Auto,49050,Petrol,325,29.4,4.0 -- Price is £44,985   Pred = £41,233.30
        # BMW,5 Series,2019,Semi-Auto,4405,Petrol,145,48.7,2.0     Price = £26,000 Pred = £27,077.49 
        return y_pred[0]

class forestRegression():

    def __init__(self, numTrees = 3, minSample = 3, maxDepth = 93):
        self.numTrees = numTrees
        self.minSamples = minSample
        self.maxDepth = maxDepth
        self.decisionTree = []
        
    
    def fit(self, X, y):
        if len(self.decisionTree) > 0:
            self.decisionTree= []
            
        num_built = 0
        while num_built < self.numTrees:
            clf = DTRegressor(3, 93)
            _X, _y = self._sample(X, y)
            # print("\nX: ", _X)
            # print("\ny: ", _y)
            print("\nOVER HERE 1\n")
            clf.fit(_X, _y)
            print("OVER HERE 2", clf)
            self.decisionTree.append(clf)
            num_built += 1
    
    

test = randomForest()
test.UIInput("Audi","RS6","2016","Semi-Auto","49050","Petrol","325","29.4","4.0")