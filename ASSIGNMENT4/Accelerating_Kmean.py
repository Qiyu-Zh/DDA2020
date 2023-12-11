from dis import dis
import numpy as np
import pandas as pd


class Acceler_Kmeans(object):

    def __init__(self, k=2, tol=0, max_iter=300):
        self.k = k
        self.tol = tol
        self.max_iter= max_iter
    def check(self,feature,list,return_index=False):
        for i,v in enumerate(list):
            if v.all()==feature.all():
                return True if return_index==False else i
            
        return False
    def fit(self, data):
        self.centers_ = {}
        for i in range(self.k):
            self.centers_[i] = data[i]
        
        for i in range(self.max_iter):
            prev_centers = dict(self.centers_)    
            self.clf_ = {}
            for i in range(self.k):
                self.clf_[i] = []
            shortest=[0]*self.k
            l_bound = np.zeros([210,3])
            for i in range(self.k):
                for j in range(self.k):
                    if j==i:
                        continue      
                    else:
                        distance=np.inf 
                        print(np.linalg.norm(self.centers_[i]-self.centers_[j]))
                        distance=min(distance,np.linalg.norm(self.centers_[i]-self.centers_[j]))
 
                shortest[i]=distance

                    
            for feature in data:
                for key,item in self.clf_.items():
                    print(self.check(feature,item),end='')
                    if self.check(feature,item):
                        
                        if 2*np.linalg.norm(feature - self.centers_[key])>shortest[key]:
                            item.pop(self.check(feature,item,return_index=True))
                            distances = []
                            for center in self.centers_:
                                distances.append(np.linalg.norm(feature - self.centers_[center]))
                            classification = distances.index(min(distances))
                            self.clf_[classification].append(feature)
                            break
                else:
                    distances = []
                    for center in self.centers_:
                        distances.append(np.linalg.norm(feature - self.centers_[center]))
                    classification = distances.index(min(distances))
                    self.clf_[classification].append(feature)
                
            
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
train=pd.read_csv('seeds_dataset.txt',header = None,sep="\t+",engine='python')
train=train.drop(columns=[7])
Accelerate_Kmean=Acceler_Kmeans(k=3)
Accelerate_Kmean.fit(np.array(train))


         