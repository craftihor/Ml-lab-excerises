import itertools

import numpy as np 
import pandas as pd

data = pd.read_csv("data.csv")
concepts = np.array(data.iloc[:,0:-1])
print("\nInstances are:\n",concepts)
target = np.array(data.iloc[:,-1])
print("\nTarget Values are: ",target)

def learn(concepts,target,vspace): 
    i = 0
    while i < len(concepts):
        h = concepts[i]
        if target[i] == "yes":
            k = 0
            while k < len(vspace):
                x = vspace[k]
                for j in range(2) : 
                    if (x[j] != h[j]) and (x[j] != "?") :                    
                        vspace.remove(x)   
                        k = k - 1
                        break
                k = k + 1
        i = i + 1
    return vspace

print("\n",data)
d = concepts
targer = target
ulist = []

for i in range(len(d[1])):
    li = []
    for j in range(len(d)):
        if d[j][i] not in li:
            li.append(d[j][i])
    li.append('?')
    ulist.append(li.copy())
    li.clear()

vspace = list(itertools.product(*ulist))
print("\nIntial vspace:\n",(vspace))

vspace = learn(concepts,target,vspace)
print("\nFinal vspace:\n",(vspace))