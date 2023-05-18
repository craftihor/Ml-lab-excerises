import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

class Linear_regression:
    def __init__(self):
        self.m = 0
        self.c = 0
    
    def train(self, X, Y):
        x_mean = np.mean(X)
        y_mean = np.mean(Y)

        n = len(X)

        numerator = 0
        denominator = 0
        for i in range(n):
            numerator += (X[i] - x_mean) * (Y[i] - y_mean)
            denominator += (X[i] - x_mean) ** 2
            
        self.m = numerator / denominator
        self.c = y_mean - (self.m * x_mean)
        
    def predict(self, x):
        return ((self.m * x) + self.c)

    def __str__(self):
        return "y = " + str(round(lr.m, 2)) + " * x + " + str(round(lr.c, 2))

dataset = pd.read_csv('sales.csv')

X = dataset['Week'].values
Y = dataset['Sales(thousands)'].values
lr = Linear_regression()
lr.train(X, Y)

print("The linear equation for the data: ", lr)
print()
print("Prediction for week 7:", lr.predict(7)) 