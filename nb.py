import numpy as np
import pandas as pd
 
data = pd.read_csv("climate.csv")
print("dataset:\n", data)
arr = np.array(data)
 
print("numpy array: \n", arr)
 
x = arr[:, :-1]
print(x) 
 
y = arr[:, -1]
print(y)
 
iters = [arr[:, x] for x in range(4)]
print("Iters:\n", iters)
 
attributeLabels = [list(set(itr)) for itr in iters]
print("Attribute Labels:\n", attributeLabels)
 
def createTable(iters, attributeLabels):
    matrix = []
    
    for itr, atr in zip(iters, attributeLabels):
        mtx = {x : [0 for y in range(2)] for x in atr}
        for el in range(len(itr)):
            for lb in atr:
                if itr[el] == lb:
                    if y[el] == "Yes":
                        mtx[lb][0] += 1
                    else:
                        mtx[lb][1] += 1
        matrix.append(mtx)
    return matrix
 
 
mtcs = createTable(iters, attributeLabels)
 
targets = [0, 0]
for el in y:
    if el == "Yes":
        targets[0] += 1
    else:
        targets[1] += 1
 
targets = [target / sum(targets) for target in targets]
 
today = ["Overcast", "Hot", "Normal", False]
 
def probability(matrix, target):
    probs = [1, 1]
    for index in range(len(matrix)):
        yescount = matrix[index][today[index]][0]
        nocount = matrix[index][today[index]][1]
        yestotal = sum([matrix[index][x][0] for x in matrix[index]] )
        nototal = sum([matrix[index][x][1] for x in matrix[index]])
        currentprobability = [yescount / yestotal , nocount / nototal]
        probs = [prob * currentProb for prob, currentProb in zip(probs, currentprobability)]
    probs = [prob * trg for prob, trg in zip(probs, target)]
    return probs
 
answer = probability(mtcs, targets)
print("\n")
print(answer)