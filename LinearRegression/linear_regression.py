import numpy as np


class LinearRegression:

    def __init__(self, lr=0.01, n_iters=1000):
        self.lr = lr
        self.n_iters = n_iters
        self.weights = None
        self.bias = None

    def fit(self, X, y):
        # Init parameters
        n_samples, n_features = X.shape
        self.weights = np.zeros(n_features)
        self.bias = 0

        # Gradient Decent
        for _ in range(self.n_iters):
            # Y-hat / Approximate
            y_predict = np.dot(X, self.weights) + self.bias

            # Derivatives
            dw = (1/n_samples) * np.dot(X.T, (y_predict - y))
            db = (1/n_samples) * np.sum(y_predict - y)

            # Update
            self.weights -= self.lr * dw
            self.bias -= self.lr * db

    def predict(self, X):
        # Return the approximate (y-hat)
        return np.dot(X, self.weights) + self.bias
