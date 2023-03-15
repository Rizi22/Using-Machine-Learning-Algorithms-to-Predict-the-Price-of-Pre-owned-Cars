import numpy as np
from decisionTree import decisionTree

class randomForest():

    def __init__(self, numTrees = 31, minSample = 6, maxDepth = 90, random_state = 0):
        self.numTrees = numTrees
        self.minSamples = minSample
        self.maxDepth = maxDepth
        self.decisionTree = []
        self.random_state = random_state
        
    def _sample(self, X, y, state):
        sampleNumb, featuresNumb = X.shape
        samples = np.random.RandomState(state).choice(a = sampleNumb, size = sampleNumb, replace = True)
        return X[samples], y[samples]
        
    def fit(self, X, y):
        if len(self.decisionTree) > 0:
            self.decisionTree= []

        num_built = 0
        for i in range(self.numTrees):
            
            try:
                DT = decisionTree(minSamples = self.minSamples, maxDepth = self.maxDepth)
                _X, _y = self._sample(X, y, self.random_state + i)
                DT.fit(_X, _y)
                self.decisionTree.append(DT)
                num_built += 1
                print("NUMBER BUILT: ", num_built)
            except Exception as e:
                print("ERROR: ", e)
                continue
    
    def predict(self, X):
        y = []
        for tree in self.decisionTree:
            y.append(tree.predict(X))
        y = np.swapaxes(a = y, axis1 = 0, axis2 = 1) 
        predictions = []
        for preds in y:
            predictions.append(np.mean(preds))
        return predictions