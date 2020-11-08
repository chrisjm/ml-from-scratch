import numpy as np
from sklearn.model_selection import train_test_split
from sklearn import datasets
import matplotlib.pyplot as plt

from logistic_regression import LogisticRegression

bc = datasets.load_breast_cancer()
print(type(bc))
X, y = bc.data, bc.target

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42)


def accuracy(y_true, y_pred):
    return np.sum(y_true == y_pred) / len(y_true)


regressor = LogisticRegression(lr=0.001, n_iters=1000)
regressor.fit(X_train, y_train)
predictions = regressor.predict(X_test)

print(f"LR accuracy: {accuracy(y_test, predictions)}")
