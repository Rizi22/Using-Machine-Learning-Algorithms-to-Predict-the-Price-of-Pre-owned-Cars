import numpy as np 

class linearRegression():
    def __init__(self,  learning_rate = 0.01, iterations = 1000):
        self.learning_rate = learning_rate
        self.iterations = iterations
        self.weights = None
        
    def fit(self, X_train, Y_train):
        # Add a column of ones to X for the intercept term
        X = np.insert(X_train, 0, 1, axis=1)
        
        # Initialize the weights to zero
        self.weights = np.zeros(X.shape[1])
        
        # Performs gradient descent for the specified number of iterations
        for i in range(self.iterations):
            Y_pred = np.dot(X, self.weights)
            error = Y_train - Y_pred
            gradient = - (2 * (X.T).dot(error)) / X.shape[0]
            self.weights = self.weights - self.learning_rate * gradient
        
        return self
        
    def predict(self, X_test):
        # Add a column of ones to X for the intercept term
        X_test = np.insert(X_test, 0, 1, axis=1)
        return np.dot(X_test, self.weights)