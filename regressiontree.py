import pandas as pd
import numpy as np

data = pd.read_csv("Job_Offer.csv")

class Node:
    def __init__(self, data):
        self.data = data
        self.children = []
    def recur(self, s, depth=0):
        if(self != None):
           s = depth * " " + "->" + self.data + "\n"
           for i in self.children:
                s += i.recur(s, depth+1)
           return s
        else:
            return ""
    
    def __str__(self):
        s = ""
        if(self == None):
            s = s + "Empty Tree"
        else:
           s = self.recur(s)
        return s
    
def find_best(data):
    totsd = np.std(data['Job_Offer'])
    maxrsd = 0.0
    best = None
    attributes = list(data.columns)
    attributes.pop()
    for i in attributes:
        u = np.unique(data[i])
        ts = 0.0
        for j in u:
            temp = data[data[i] == j]
            ts = ts + (len(temp)/len(data) * (np.std(temp['Job_Offer'])))
        rsd = totsd - ts
        if rsd > maxrsd:
            maxrsd = rsd
            best = i
    return best


def rtree(data, cnode = None, depth=0):
    best = find_best(data)
    if(depth == 3 or best == None):
        tn = Node("Value: " + str(np.mean(data['Job_Offer'])))
        cnode.children.append(tn)
    if(best != None):
        if cnode == None: 
            cnode = Node(best)
        else:
            tn = Node(best)
            cnode.children.append(tn)
            cnode = tn
        u = np.unique(data[best])
        for i in u:
            tn = Node(i)
            rtree(data[data[best] == i], tn, depth+1)
            cnode.children.append(tn)
    return cnode
        

tree = rtree(data, None)

print("Regression tree: ")
print(tree)
            