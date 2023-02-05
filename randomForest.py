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
        myForest = forestRegression()  
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
        print("\n Predicted price for your car is: £", y_pred)

        print("\n ***Predicted in", time.time() - start,"seconds***")

        # RS6,2016,Semi-Auto,49050,Petrol,325,29.4,4.0 -- Price is £44,985   Pred = £41,233.30
        # BMW,5 Series,2019,Semi-Auto,4405,Petrol,145,48.7,2.0     Price = £26,000 Pred = £27,077.49 
        return y_pred

class forestRegression():

    def __init__(self, numTrees = 14, minSample = 5, maxDepth = 5):
        self.numTrees = numTrees
        self.minSamples = minSample
        self.maxDepth = maxDepth
        self.decisionTree = []
        
    @staticmethod
    def _sample(X, y):
        n_rows, n_cols = X.shape
        samples = np.random.RandomState(601).choice(a = n_rows, size = n_rows, replace = True)
        # samples =rnd.choice(a = n_rows, size = n_rows, replace = True)
        return X[samples], y[samples]
        
    def fit(self, X, y):
        if len(self.decisionTree) > 0:
            self.decisionTree= []
            
        num_built = 0
        while num_built < self.numTrees:
            print("NUMBER BUILT: ", num_built)
            try:
                clf = DTRegressor(minSamples = self.minSamples, maxDepth = self.maxDepth) ##try 3, then 1
                _X, _y = self._sample(X, y)
                clf.fit(_X, _y)
                self.decisionTree.append(clf)
                num_built += 1
            except Exception as e:
                continue



        # for i in range(self.numTrees):
        #     # Randomly sample the data for each tree
        #     sampleIndices = np.random.choice(X.shape[0], size = X.shape[0], replace = True)
        #     X_sample = X[sampleIndices,:]
        #     Y_sample = y[sampleIndices]
            
        #     # Fit a decision tree to the sample data
        #     treeModel = DTRegressor(minSamples = self.minSamples, maxDepth = self.maxDepth)
        #     treeModel.fit(X_sample, Y_sample)
        #     self.decisionTree.append(treeModel)
    
    def predict(self, X):
        y = []
        for tree in self.decisionTree:
            y.append(tree.predict(X))


        # Y_pred = np.zeros((X.shape[0], 1))
        # for treeModel in self.treeModels:
        #     Y_pred += treeModel.predict(X)
        # Y_pred /= self.numTrees
        # return Y_pred

        # print("\nBEFORE:", y)
        # y = np.swapaxes(a=y, axis1=0, axis2=1)
        # print("\nAFTER:", y)
        
        # predictions = []
        # for preds in y:
        #     counter = Counter(pred)
        #     test.append(X)
        #     predictions.append(counter.most_common(1)[0][0])
        # return predictions
        # print("HERE: ", np.mean(y))
        return np.mean(y)

# print("TEST1")
test = randomForest()
# print("TEST2")
# test.UIInput("Audi","RS6","2016","Semi-Auto","49050","Petrol","325","29.4","4.0")
test.UIInput("BMW","5 Series","2019","Semi-Auto","4405","Petrol","145","48.7","2.0")
# print("TEST3")