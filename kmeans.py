import math
class kmeans:
    def __init__(self, k, n):
        self.k=k
        self.centers= []
        self.clusters=[]
        self.n = n
    def distance(n,x1,x2):
        k = 0
        for i in range(n):
            k+=(x1[i]-x2[i])**2
        return math.sqrt(k)
       
   
    def cluster(self, data):
        self.clusters = [0] * len(data)
        for i in range(self.k):
            self.centers.append(list(data[i]))
        prev = None
        while(True):
            for i in range(len(data)):
                ind = 0
                mind = 10000
                for j in range(len(self.centers)):
                    d=kmeans.distance(self.n,self.centers[j],data[i])
                    if(d < mind):
                        mind = d
                        ind = j
                self.clusters[i] = ind
            if(prev == self.clusters):
                break
            prev = self.clusters
            cc=[0]*self.k
            for i in range(self.k):
                for j in range(self.n):
                    self.centers[i][j] = 0
            for i in range(len(self.clusters)):
                cc[self.clusters[i]] += 1
                for j in range(self.n):
                    self.centers[self.clusters[i]][j] += data[i][j]
            for i in range(self.k):
                for j in range(self.n):
                    self.centers[i][j] = self.centers[i][j]/cc[i]
        
   

data = [[-0.1,0.2,0.3],[0.5,0.2,0.3],[0.1,0.2,0.3],[0.5,-0.5,-0.3]]
k = kmeans(2,3)
k.cluster(data)

print("The centroids after the final iteration")
for i in range(k.k):
    print(k.centers[i])

print("\nMembership of the points")
print("        Point             Centroid")
for i in range(len(data)):
    print(i," ",data[i]," ",k.centers[k.clusters[i]])