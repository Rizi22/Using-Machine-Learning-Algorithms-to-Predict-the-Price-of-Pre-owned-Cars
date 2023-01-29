import numpy as np
from collections import Counter
from DT import DT
decisionTree = DT()

class randomForest:

    def __init__(self, numTrees = 25, minSample = 3, maxDepth = 93):
        self.numTrees = numTrees
        self.minSamples = minSample
        self.maxDepth = maxDepth
        self.dt = []
        
    