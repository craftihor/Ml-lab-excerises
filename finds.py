import pandas as pd
import numpy as np

data = pd.read_csv("data.csv")
print("The data is:\n",data)

arr = np.array(data)[:,:-1]

res = np.array(data)[:,-1]

def finds(arr, res):
    for i in range(0, len(res)):
        if(res[i] == 'Yes'):
            specific_hypothesis = arr[i].copy()
            break
    for i in range(0, len(res)):
        if(res[i] == 'Yes'):
            for j in range(0, len(arr[0])):
                if(specific_hypothesis[j] != arr[i][j]):
                    specific_hypothesis[j] = '?'
    return specific_hypothesis

print("\nThe final hypothesis is:", finds(arr, res))