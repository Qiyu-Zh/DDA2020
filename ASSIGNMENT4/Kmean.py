import numpy as np
import pandas as pd



class Kmeans(object):

    def __init__(self, k=2, tol=0, max_iter=300):
        self.k = k
        self.tol = tol
        self.max_iter= max_iter

    def fit(self, data):
        max_val=np.max(data,axis=0)
        min_val=np.max(data,axis=0)
        
        self.centers_ = {}
        for i in range(self.k):
            self.centers_[i] = data[i]

        for _ in range(self.max_iter):
            self.clf_ = {}
            for i in range(self.k):
                self.clf_[i] = []
            for feature in data:
                distances = []
                for center in self.centers_:
                    distances.append(np.linalg.norm(feature - self.centers_[center]))
                classification = distances.index(min(distances))
                self.clf_[classification].append(feature)
            prev_centers = dict(self.centers_)
            for c in self.clf_:
                self.centers_[c] = np.average(self.clf_[c], axis=0)

            optimized = True
            for center in self.centers_:
                org_centers = prev_centers[center]
                cur_centers = self.centers_[center]
                if np.sum(cur_centers - org_centers)**2 > self.tol:
                    optimized = False
            if optimized==True:
                break
    def predict(self, p_data):
        distances = [np.linalg.norm(p_data - self.centers_[center]) for center in self.centers_]
        index = distances.index(min(distances))
        return index
    def get_Silhouette_coefficient(self,x):
        
        a,b=0,0
        for p_data in x:
            distances = [np.linalg.norm(p_data - self.centers_[center]) for center in self.centers_]
            smallest,smallest_index=np.inf,0
            secondsmallest,secondsmallest_index=np.inf,0
            for i,v in enumerate(distances):
                if v<smallest:
                    secondsmallest=smallest
                    secondsmallest_index=smallest_index
                    smallest=v
                    smallest_index=i
                if secondsmallest>v>smallest:
                    secondsmallest=v
                    secondsmallest_index=i
            

            for i in self.clf_[smallest_index]:
                a+=np.linalg.norm(p_data-i)
            a/=len(self.clf_[smallest_index])
            
            for i in self.clf_[secondsmallest_index]:
                b+=np.linalg.norm(p_data-i)
            b/=len(self.clf_[secondsmallest_index])
            
        return (b-a)/max(b,a)       
                
    def rand_index(self, y_true, y_pred):
        n = len(y_true)
        a, b = 0, 0
        for i in range(n):
            for j in range(i+1, n):
                if (y_true[i] == y_true[j]) & (y_pred[i] == y_pred[j]):
                    a +=1
                elif (y_true[i] != y_true[j]) & (y_pred[i] != y_pred[j]):
                    b +=1
                else:
                    pass
        RI = (a + b) / (n*(n-1)/2)
        return RI

            
        

         

train=pd.read_csv('seeds_dataset.txt',header = None,sep="\t+",engine='python').drop(columns=[7])
label=pd.read_csv('seeds_dataset.txt',header = None,sep="\t+",engine='python')[7]
c=np.array(label)
b=np.array(train)
Accelerate_Kmean=Kmeans(k=3)
Accelerate_Kmean.fit(b)
predict=np.array([0]*len(b))
for i,v in enumerate(b):
    predict[i]=Accelerate_Kmean.predict(v)
    
print(Accelerate_Kmean.rand_index(label,predict))


