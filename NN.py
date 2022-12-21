import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from math import sqrt
from sklearn.metrics import mean_squared_error 
from sklearn.preprocessing import LabelEncoder

class NN():

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

        file = file.head(1000) # Limits dataset size

        X = file.drop(columns = ['price'])
        Y = file.price
        
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, random_state = 601)
        
        self.scaler.fit(X_train)

        X_train = self.scaler.transform(X_train)
        X_test = self.scaler.transform(X_test)

        return  X_train, X_test, Y_train, Y_test

    def eucDistance(self, variable1, variable2):
        distance = 0
        for i in range(len(variable2)):
            distance += (variable1[i] - variable2[i])**2
        return np.sqrt(distance)

    def kNN(self, train, testRow, yTrain, num):
        distance = list() #Stores distance of each point
        for i in range(len(train)): 
            dist = self.eucDistance(train[i], testRow) #sends points to work out distance
            distance.append((train[i], dist, yTrain.iloc[i])) 
        distance = self.sort(distance)
        kNeighbours = list() #list to store K amount of neighbour results
        for i in range(num):
            kNeighbours.append((testRow, distance[i][1], distance[i][2]))
        return kNeighbours 

    def sort(self, dist):
        for i in range(0, len(dist)):
            for j in range(0, len(dist) - i - 1):
                if (dist[j][1] > dist[j + 1][1]):
                    temp = dist[j]
                    dist[j] = dist[j + 1]
                    dist[j + 1] = temp
        return dist
    
    def predict(self, train, test, yTrain, num_neighbors):
        from statistics import mean 
        predictions = list()
        for i in range(len(test)):
            neighbour = self.kNN(train, test[i], yTrain, num_neighbors) #CHANGE
            labels = [] #Stores yTrain for each test variable
            for i in range (len(neighbour)):
                labels.append(neighbour[i][2]) #Appends yTrain
            predictions.append(mean(labels))
        return predictions
    
    def rmse(self, test, pred):
        MSE = np.square(np.subtract(test, pred)).mean()
        return sqrt(MSE)

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

    def testing(self, chooseBrand):
        self.dataset(self.userInput(chooseBrand))
        return
        
            
    def UIInput(self, chooseBrand, model, year, transmission, mileage, fuelType, tax, mpg, engineSize):
        inputPred = []
        entries = []

        X_train, X_test, Y_train, Y_test = self.dataset(self.userInput(chooseBrand))

        inputPred.append((self.modelEncoder.transform([model]))[0])
        inputPred.append(int(year))
        inputPred.append((self.transmissionEncoder.transform([transmission]))[0])
        inputPred.append(int(mileage))
        inputPred.append((self.fuelTypeEncoder.transform([fuelType]))[0])
        inputPred.append(int(tax))
        inputPred.append(float(mpg))
        inputPred.append(float(engineSize))
        entries.append(inputPred)
        inputPred = self.scaler.transform([inputPred])

        import time
        print("\n ***Predicting***")
        start = time.time()
        y_pred = self.predict(X_train, inputPred, Y_train, 4)
        # {0:.2f}'.format()
        print("\n Predicted price for your car is: £", y_pred[0])

        print("\n ***Predicted in", time.time() - start,"seconds***")
        return y_pred[0]

# test = NN()
# test.UIInput("Audi","RS6","2016","Semi-Auto","49050","Petrol","325","29.4","4.0")
# Audi,RS6,2016,Semi-Auto,49050,Petrol,325,29.4,4.0    Price = £44,985 Pred:43,993 --- £44717