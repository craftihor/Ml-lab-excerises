import numpy as np
import pandas as pd

class DecisionStump():
    def __init__(self):
        self.polarity = 1
        self.feature_idx = None
        self.threshold = None
        self.alpha = None

    def predict(self, X):
        n_samples = X.shape[0]
        X_column = X[:, self.feature_idx]
        predictions = np.ones(n_samples)
        if self.polarity == 1:
            predictions[X_column < self.threshold] = -1
        else:
            predictions[X_column > self.threshold] = -1

        return predictions


class Adaboost():
    def __init__(self, n_clf=5):
        self.n_clf = n_clf

    def fit(self, X, y):
        n_samples, n_features = X.shape
        w = np.full(n_samples, (1 / n_samples))
        self.clfs = []
        for _ in range(self.n_clf):
            clf = DecisionStump()
            min_error = float('inf')
            for feature_i in range(n_features):
                X_column = X[:, feature_i]
                thresholds = np.unique(X_column)
                for threshold in thresholds:
                    p = 1
                    predictions = np.ones(n_samples)
                    predictions[X_column < threshold] = -1
                    misclassified = w[y != predictions]
                    error = sum(misclassified)
                    if error > 0.5:
                        error = 1 - error
                        p = -1
                    if error < min_error:
                        clf.polarity = p
                        clf.threshold = threshold
                        clf.feature_idx = feature_i
                        min_error = error
            EPS = 1e-10
            clf.alpha = 0.5 * np.log((1.0 - min_error + EPS) / (min_error + EPS))
            predictions = clf.predict(X)
            w *= np.exp(-clf.alpha * y * predictions)
            w /= np.sum(w)
            self.clfs.append(clf)
    
    def predict(self, X):
        clf_preds = [clf.alpha * clf.predict(X) for clf in self.clfs]
        y_pred = np.sum(clf_preds, axis=0)
        y_pred = np.sign(y_pred)
        return y_pred

data = pd.read_csv('data.csv')

print("\nDataset:")
print(data,"\n")
inputmap = {'Yes': 1, 'No': 0, 'Good': 1, 'Moderate': 0, 'Average': 0}
ouputmap = {'Yes': 1, 'No': -1}

y = data.pop('Job_Offer')
X = data

X = X.replace(inputmap)
y = y.replace(ouputmap)

X = X.values
y = y.values

adaboost = Adaboost(n_clf=5)
adaboost.fit(X, y)

y_pred = adaboost.predict(X)
print("Map: ",inputmap)
print("\nPredictions of the adaboost classifier: ")
print(list(data.columns), "Job_Offer")
for i in range(len(y_pred)):
    print(X[i]," = ",end="")
    if(y_pred[i] == 1):
        print("Yes")
    else:
        print("No")