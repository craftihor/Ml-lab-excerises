import pandas as pd
import numpy as np
import math
import itertools


data = pd.read_csv("Job_Offer.csv")
output = "Job_Offer"
feat = list(data.columns)
feat.pop()

class Node:
    def __init__(self, data):
        self.data = data
        self.children = []
        
    def recur(self, s, depth=0):
        if(self != None):
           s = depth * " " + "->" + str(self.data) + "\n"
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

def gini(data):
    gini = 0.0
    t = len(data)
    p = len(data[data[output] == "Yes"])
    n = len(data[data[output] == "No"])
    if t != 0:
        gini = 1 - (math.pow(p/t, 2)) - (math.pow(n/t, 2))
    return gini
    
def find_best(data, feat):
    bg = 0.0
    bf = []
    bi = None
    tot_gini = gini(data) 
    for i in feat:
        t = data[i]
        uniq = np.unique(t)
        if(len(uniq) > 2):
            for k in range(1, len(uniq)-1):
                x = list(itertools.combinations(uniq, len(uniq) - k))
                for j in x:
                    y = list(set(uniq) - set(j))
                    gx = data[data[i].isin(j)]
                    gy = data[data[i].isin(y)]
                    ginix = gini(gx)
                    giniy = gini(gy)
                    tg = (len(gx) / (len(gx) + len(gy))) * ginix
                    tg += (len(gy) / (len(gx) + len(gy))) * giniy
                    tg = tot_gini - tg
                    if bg < tg:
                        bg = tg
                        bf = []
                        bi = i
                        bf.append(j)
                        bf.append(y)
        else:
            t = 0.0
            for j in uniq:
                x = data[data[i] == j]
                t += (len(x) / len(data[i])) * gini(x)
            t = tot_gini - t
            if t > bg:
                bg = t
                bf = []
                bf.append(uniq[0])
                bf.append(uniq[1])
                bi = i
    return bi, bf
    
def cart(data, feat, cnode = None, depth=0):
    best, cond = find_best(data, feat)
    uniq = np.unique(data[output])
    if(len(uniq) == 1):
        tn = Node("Value: " + uniq)
        cnode.children.append(tn)
    if(best != None):
        feat.remove(best)
        if cnode == None: 
            cnode = Node(best)
        else:
            tn = Node(best)
            cnode.children.append(tn)
            cnode = tn
        u = cond
        for i in u:
            tn = Node(i)
            cart(data[data[best].isin(i)], feat, tn, depth+1)
            cnode.children.append(tn)
    return cnode
                

tree = cart(data, feat)
print("CART Tree:")
print(tree)