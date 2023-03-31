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
        
    # Compute the information gain for a given split    
    def infoGain(self, parent, leftNode, rightNode):
        # Compute the weight of the left and right node
        leftWeight = len(leftNode) / len(parent)
        rightWeight = len(rightNode) / len(parent)
        
        # Compute the information gain based on the variance of the parent and children nodes
        informationGain = np.var(parent) - (leftWeight * np.var(leftNode) + rightWeight * np.var(rightNode))

        return informationGain
        
    # Split the training data into left and right branches based on a threshold
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
    
    # Find the best split for a given feature
    def bestSplit(self, trainingSet, X):
        bestSplitt = {} 
        biggestGain = -1

        # Loop through each feature and each threshold value to find the best spli
        for feature in range(X.shape[1]): 
            featureValues = []
            for i in range(len(trainingSet)):
                featureValues.append(trainingSet[i, feature])
            thresholds = np.unique(featureValues)
            for j in thresholds:
                # Split the node into two sub-trees
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
                        # Update the best split if the gain is larger than the previous best gain
                        bestSplitt["feature"] = feature
                        bestSplitt["limit"] = j
                        bestSplitt["leftSide"] = leftSide
                        bestSplitt["rightSide"] = rightSide
                        bestSplitt["gain"] = currentGain
                        biggestGain = currentGain
                
        return bestSplitt
   
    # Build the decision tree recursively
    def treeBuild(self, trainingSet, currentDepth = 0):
        
        # Split training data into features and labels
        X = trainingSet[:,:-1] # Everything but the last value
        Y = []
        for i in range(len(trainingSet)):
            Y.append(trainingSet[i, -1]) # only the last value
        
        # Recursively build the tree until the stopping condition is met
        if X.shape[0] >= self.minSamples and currentDepth <= self.maxDepth:
            bestSplitNode = self.bestSplit(trainingSet, X)
            
            if "gain" in bestSplitNode and bestSplitNode["gain"] > 0:
                leftTree = self.treeBuild(bestSplitNode["leftSide"], currentDepth + 1)
                rightTree = self.treeBuild(bestSplitNode["rightSide"], currentDepth + 1)
                node = Node(bestSplitNode["feature"], bestSplitNode["limit"], leftTree, rightTree, bestSplitNode["gain"])

                return node
        
        # Create a leaf node with the average value of the training set
        leafValue = np.mean(Y) #calculates mean of leaf nodes
        val = Node(leaf = leafValue)
        return val
    
    # Recursive function predict the label for a test row
    def predictionLoop(self, testRow, root):
        if root.leaf != None: # Leaf node
            return root.leaf
        
        featureVal = testRow[root.feature]
        if featureVal <= root.limit:
            return self.predictionLoop(testRow, root.leftSide)
        else:
            return self.predictionLoop(testRow, root.rightSide)
        
   
    # Function to predict the labels for a set of test rows
    def predict(self, xTest):
        predictions = []
        for row in xTest:
            predictions.append(self.predictionLoop(row, self.root)) 
        return predictions

        
    # Function to train the decision tree and set the root node
    def fit(self, X, Y):
        trainingSet = np.concatenate((X, Y), axis=1) # Join features and labels
        self.root = self.treeBuild(trainingSet)