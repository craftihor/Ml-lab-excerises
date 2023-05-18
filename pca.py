import numpy as np 
import pandas as pd
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_csv("iris.csv")
print(df.head())
y = df.pop("species").values
X = df.values
X_scaled = StandardScaler().fit_transform(X)

features = X_scaled.T
cov_matrix = np.cov(features)
values, vectors = np.linalg.eig(cov_matrix)

explained_variances = []
for i in range(len(values)):
    explained_variances.append(values[i] / np.sum(values))
 
print(np.sum(explained_variances),"\n",explained_variances)

projected_1 = X_scaled.dot(vectors.T[0])
projected_2 = X_scaled.dot(vectors.T[1])
res = pd.DataFrame(projected_1, columns=["PC1"])
res["PC2"] = projected_2
res["Y"] = y
print(res.head())


#plt.figure(figsize=(20, 10))
#sns.scatterplot(res["PC1"], [0] * len(res), hue=res["Y"], s=200)

#plt.figure(figsize=(20, 10))
#sns.scatterplot(x = res["PC1"], y = res["PC2"], hue=res["Y"], s=100)