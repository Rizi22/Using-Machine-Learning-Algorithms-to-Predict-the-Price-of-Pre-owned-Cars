import numpy as np
from decisionTree import decisionTree

class randomForest():

    # Initialize the model with hyperparameters
    def __init__(self, numTrees = 31, minSample = 6, maxDepth = 90, random_state = 0):
        self.numTrees = numTrees # Number of trees in the forest
        self.minSamples = minSample # Minimum number of samples required to split an internal node
        self.maxDepth = maxDepth # Maximum depth of the decision trees
        self.decisionTree = [] # List to hold decision trees of the fores
        self.random_state = random_state # Random state used to generate bootstrap samples
        
    # Function to generate bootstrap samples 
    def bootstrapSample(self, X, y, state):
        sampleNumb, featuresNumb = X.shape
        samples = np.random.RandomState(state).choice(a = sampleNumb, size = sampleNumb, replace = True)
        return X[samples], y[samples]

    # Method to fit the random forest on training data   
    def fit(self, X, y):
        if len(self.decisionTree) > 0:
            self.decisionTree= []

        num_built = 0
        # Loop over the number of trees to build
        for i in range(self.numTrees):
            
            try:
                DT = decisionTree(minSamples = self.minSamples, maxDepth = self.maxDepth) # Create a decision tree object
                X, y = self.bootstrapSample(X, y, self.random_state + i) # Generate bootstrap sample of data
                DT.fit(X, y) # Fit the decision tree on bootstrap sample
                self.decisionTree.append(DT) # Add the decision tree to the forest
                num_built += 1
                # print("NUMBER BUILT: ", num_built)

            except Exception as e:
                print("ERROR: ", e) # Handle any exception that occurs during building a decision tree
                continue
    
    # Method to predict the target values on new data using the random forest
    def predict(self, X):
        y = []
        
        # Loop over the decision trees of the forest to get their predictions
        for tree in self.decisionTree:
            y.append(tree.predict(X))
        y = np.swapaxes(a = y, axis1 = 0, axis2 = 1) 
        predictions = []

        # Loop over the predictions of each row and compute their mean as the final prediction
        for preds in y:
            predictions.append(np.mean(preds))
        return predictions