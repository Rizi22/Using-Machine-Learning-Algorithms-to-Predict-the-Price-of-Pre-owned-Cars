import numpy as np 

class linearRegression() :
      
    def __init__(self, learning_rate, iterations) :
        self.learning_rate = learning_rate
        self.iterations = iterations
        
              
    def fit(self, X, Y) :
        self.sampleNumb, self.featuresNumb = X.shape 
        self.weight = np.zeros(self.featuresNumb)  
        self.bias = 0  
        self.X = X 
        self.Y = Y
                  
        for i in range(self.iterations) :             
            self.weights()             
        return self
      
    def weights(self) :
        Y_pred = self.predict(self.X)
        weightGradient = - (2 * (self.X.T).dot(self.Y - Y_pred )) / self.sampleNumb
        biasGradient = - 2 * np.sum(self.Y - Y_pred) / self.sampleNumb
        self.weight = self.weight - self.learning_rate * weightGradient
        self.bias = self.bias - self.learning_rate * biasGradient
        return self
      
    def predict(self, X) :
        return X.dot(self.weight) + self.bias