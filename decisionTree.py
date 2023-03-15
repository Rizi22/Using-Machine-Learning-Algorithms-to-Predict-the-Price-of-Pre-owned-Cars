import numpy as np

# Node class to initialise instances of each 
class Node():
    
    def __init__(self, feature = None, limit = None, leftSide = None, rightSide = None, gain = None, leaf = None):
        
        self.feature = feature
        self.limit = limit
        self.leftSide = leftSide
        self.rightSide = rightSide
        self.gain = gain
        self.leaf = leaf 

class decisionTree():
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