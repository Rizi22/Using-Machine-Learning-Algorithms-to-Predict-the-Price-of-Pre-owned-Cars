import numpy as np
from math import sqrt

class nearestNeighbour():

    def eucDistance(self, variable1, variable2):
        distance = 0
        for i in range(len(variable2)):
            distance += (variable1[i] - variable2[i])**2
        return np.sqrt(distance)

    def kNN(self, train, testRow, yTrain, num):
        distance = list() #Stores distance of each point
        for i in range(len(train)): 
            dist = self.eucDistance(train[i], testRow) # Work out distance to point
            distance.append((train[i], dist, yTrain.iloc[i])) 
        distance = self.sort(distance)
        kNeighbours = list() # list to store K amount of neighbour results
        for i in range(num):
            kNeighbours.append((testRow, distance[i][1], distance[i][2]))
        return kNeighbours 

    # Sorts the distance list in ascending order
    def sort(self, dist):
        for i in range(0, len(dist)):
            for j in range(0, len(dist) - i - 1):
                if (dist[j][1] > dist[j + 1][1]):
                    temp = dist[j]
                    dist[j] = dist[j + 1]
                    dist[j + 1] = temp
        return dist
    
    # Predicts the value of the test variable
    def predict(self, train, test, yTrain, num_neighbors):
        from statistics import mean 
        predictions = list()
        for i in range(len(test)):
            neighbour = self.kNN(train, test[i], yTrain, num_neighbors) 
            labels = [] # Stores yTrain for each test variable
            for i in range (len(neighbour)):
                labels.append(neighbour[i][2]) # Appends yTrain
            predictions.append(mean(labels))
        return predictions
    
    # Calculates the root mean squared error
    def rmse(self, test, pred):
        MSE = np.square(np.subtract(test, pred)).mean()
        return sqrt(MSE)