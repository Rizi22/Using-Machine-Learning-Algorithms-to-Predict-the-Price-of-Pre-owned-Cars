import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import time

# class decisionTree():
    
#     def __init__(self):
#         self.modelEncoder = LabelEncoder()
#         self.transmissionEncoder = LabelEncoder()
#         self.fuelTypeEncoder = LabelEncoder()
    
#     def dataset(self, brand):

#         file = pd.read_csv(brand, quotechar='"', skipinitialspace=True)

#         for i in ['year']:
#             q75,q25 = np.percentile(file.loc[:,i],[75,25])
#             IQR = q75-q25
        
#             maxQ = q75+(1.5*IQR)
#             minQ = q25-(1.5*IQR)
        
#             file.loc[file[i] < minQ, i] = np.nan
#             file.loc[file[i] > maxQ, i] = np.nan

#         file = file.dropna(axis = 0)

#         self.modelEncoder.fit(file["model"])
#         file["model"] = self.modelEncoder.transform(file["model"])

#         self.transmissionEncoder.fit(file["transmission"])
#         file["transmission"] = self.transmissionEncoder.transform(file["transmission"])

#         self.fuelTypeEncoder.fit(file["fuelType"])
#         file["fuelType"] = self.fuelTypeEncoder.transform(file["fuelType"])

#         file = file.head(5000)

#         X = file.drop(['price'], axis = 1).to_numpy()
#         Y = file['price'].values.reshape(-1,1)

#         X_train, X_test, Y_train, Y_test = train_test_split(X, Y, random_state = 601)
        
#         return  X_train, X_test, Y_train, Y_test
    
#     def testing(self, chooseBrand):
#         self.dataset(self.userInput(chooseBrand))
#         return

#     def userInput(self, chooseBrand):

#         match chooseBrand:
#             case "Audi":
#                 return "UKUsedCarDataSet/audi.csv"
#             case "BMW":
#                 return "UKUsedCarDataSet/bmw.csv"
#             case "Ford":
#                 return "UKUsedCarDataSet/ford.csv"
#             case "Hyundai":
#                 return "UKUsedCarDataSet/hyundi.csv"
#             case "Mercedes":
#                 return "UKUsedCarDataSet/merc.csv"
#             case "Skoda":
#                 return "UKUsedCarDataSet/skoda.csv"
#             case "Toyota":
#                 return "UKUsedCarDataSet/toyota.csv"
#             case "Vauxhall":
#                 return "UKUsedCarDataSet/vauxhall.csv"
#             case "Volkswagen":
#                 return "UKUsedCarDataSet/volkswagen.csv"
#             case _:
#                 print("Invalid input")
#                 return
        
#     def UIInput(self, chooseBrand, model, year, transmission, mileage, fuelType, tax, mpg, engineSize):

#         inputPred = []

#         X_train, X_test, Y_train, Y_test = self.dataset(self.userInput(chooseBrand))

#         inputPred.append((self.modelEncoder.transform([model]))[0])
#         inputPred.append(int(year))
#         inputPred.append((self.transmissionEncoder.transform([transmission]))[0])
#         inputPred.append(int(mileage))
#         inputPred.append((self.fuelTypeEncoder.transform([fuelType]))[0])
#         inputPred.append(int(tax))
#         inputPred.append(float(mpg))
#         inputPred.append(float(engineSize))

#         print("\n ***Training Tree Model***")
#         timer = time.time()

#         myTree = DTRegressor(3, 93)
#         myTree.fit(X_train, Y_train)

#         # print("\n ***Predicting***")
#         Y_pred = myTree.predict([inputPred])
#         print("\n Predicted price for your car is: £",round(Y_pred[0], 2))
#         print("\n ***Predicted in", time.time() - timer,"seconds***")

#         return Y_pred[0]

      
# Node class to initialise instances of each 
class Node():
    
    def __init__(self, feature = None, limit = None, leftSide = None, rightSide = None, gain = None, leaf = None):
        
        self.feature = feature
        self.limit = limit
        self.leftSide = leftSide
        self.rightSide = rightSide
        self.gain = gain
        self.leaf = leaf 

class DTRegressor():
# class decisionTree():
    def __init__(self, minSamples, maxDepth):
        self.root = None
        self.minSamples = minSamples
        self.maxDepth = maxDepth
        
    def infoGain(self, parent, leftNode, rightNode):
        leftWeight = len(leftNode) / len(parent)
        rightWeight = len(rightNode) / len(parent)
        
        information_gain = np.var(parent) - (leftWeight * np.var(leftNode) + rightWeight * np.var(rightNode))

        return information_gain
        
    def splitTree(self, trainingSet, feature, limit):
        leftBranch = []
        rightBranch = []
        for i in trainingSet:
            if i[feature] <= limit:
                leftBranch.append(i)
            else:
                rightBranch.append(i)
        rightBranch = np.array(rightBranch)
        leftBranch = np.array(leftBranch)
        return leftBranch, rightBranch
        
    def bestSplit(self, trainingSet, X):
        bestSplitt = {} 
        biggestGain = -1
        for feature in range(X.shape[1]): 
            featureValues = []
            for i in range(len(trainingSet)):
                featureValues.append(trainingSet[i, feature])
            thresholds = np.unique(featureValues)
            for j in thresholds:
                leftSide, rightSide = self.splitTree(trainingSet, feature, j) #splits node into 2 sub-trees
                if (len(leftSide) > 0 and len(rightSide) > 0 ):
                    parent = []
                    for i in range(len(trainingSet)):
                        parent.append(trainingSet[i, -1])

                    leftNode = []
                    for i in range(len(leftSide)):
                        leftNode.append(leftSide[i, -1])
                        
                    rightNode = []
                    for i in range(len(rightSide)):
                        rightNode.append(rightSide[i, -1])

                    currentGain = self.infoGain(parent, leftNode, rightNode) 
                    if currentGain > biggestGain: 

                        bestSplitt["feature"] = feature
                        bestSplitt["limit"] = j
                        bestSplitt["leftSide"] = leftSide
                        bestSplitt["rightSide"] = rightSide
                        bestSplitt["gain"] = currentGain
                        biggestGain = currentGain

        return bestSplitt
           
    def treeBuild(self, trainingSet, currentDepth = 0):

        # Split training into features and labels
        X = trainingSet[:,:-1] # everything but the last value
        Y = []
        for i in range(len(trainingSet)):
            Y.append(trainingSet[i, -1])# only the last value
        
        #iterates until this condition is met
        if X.shape[0] >= self.minSamples and currentDepth <= self.maxDepth:
            bestSplitNode = self.bestSplit(trainingSet, X)

            if "gain" in bestSplitNode and bestSplitNode["gain"] > 0:
                leftTree = self.treeBuild(bestSplitNode["leftSide"], currentDepth + 1)
                rightTree = self.treeBuild(bestSplitNode["rightSide"], currentDepth + 1)
                node = Node(bestSplitNode["feature"], bestSplitNode["limit"], leftTree, rightTree, bestSplitNode["gain"])
                
                return node
                
        leafValue = np.mean(Y) #calculates mean of leaf nodes
        val = Node(leaf = leafValue)
        return val
    
    def predictionLoop(self, testRow, root):
        if root.leaf != None:
            return root.leaf
        
        featureVal = testRow[root.feature]
        if featureVal <= root.limit:
            return self.predictionLoop(testRow, root.leftSide)
        else:
            return self.predictionLoop(testRow, root.rightSide)
        
    def predict(self, xTest):
        predictions = []
        for row in xTest:
            predictions.append(self.predictionLoop(row, self.root)) 
        return predictions
  
    def fit(self, X, Y):
        trainingSet = np.concatenate((X, Y), axis=1) #Joins training data back together
        self.root = self.treeBuild(trainingSet)

# test = decisionTree()
# test.UIInput("Audi","RS6","2016","Semi-Auto","49050","Petrol","325","29.4","4.0") #£45,492
# test.UIInput("BMW","5 Series","2019","Semi-Auto","4405","Petrol","145","48.7","2.0")