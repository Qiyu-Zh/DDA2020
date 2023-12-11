import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import multivariate_normal
import pandas as pd
class GMM(object):
    def __init__(self, n_clusters, tol=1e-4, max_iter=500):
        self.n_clusters = n_clusters
        self.max_iter = max_iter
        self.tol=tol
        self.clf_={}
        for i in range(n_clusters):
            self.clf_[i]=[]
    
    def GMM_EM(self):
        '''
        利用EM算法进行优化GMM参数的函数
        :return: 返回数据属于每个分类的概率
        '''
        loglikelyhood = 0
        oldloglikelyhood = 1
        len, dim = np.shape(self.data)
        #gammas[len*n_clusters]后验概率表示第n个样本属于第k个混合高斯的概率，类似K-Means中的rnk，第n个点是否属于第k类
        gammas = np.zeros((len, self.n_clusters))
   		#迭代
        i = 0
        while np.abs(loglikelyhood - oldloglikelyhood) > self.tol or i < self.max_iter:
            oldloglikelyhood = loglikelyhood

            for k in range(self.n_clusters):
                try:
                    prob = multivariate_normal.pdf(self.data,self.means[k],self.covars[k])
                except:
                    continue
                gammas[:,k] = prob * self.weights[k]
                gamma_sum = np.sum(gammas,axis=1)
            for k in range(self.n_clusters):
                gammas[:,k] = gammas[:,k] / gamma_sum
            # M-Step
            for k in range(self.n_clusters):
                # Nk表示样本中有多少概率属于第k个高斯分布
                Nk = np.sum(gammas[:,k])
                #更新每个高斯分布的权重 pi
                self.weights[k] = 1.0 * Nk /len
                #更新高斯分布的均值 Mu
                '''self.means[k] = (1.0/Nk)*np.sum([gammas[n][k] * self.data[n] for n in range(len)],axis=0)'''
                self.means[k] = (1.0/Nk)*np.array([np.sum(gammas[:,k] * self.data[:,d]) for d in range(dim)])
                # 更新高斯分布的协方差矩阵 Var
                xdiffs = self.data - self.means[k]
                '''self.covars[k] = (1.0/Nk) * np.sum([gammas[n][k] * xdiffs[n].reshape((dim,1)).dot(xdiffs[n].reshape((1,dim))) for n in range(len)],axis=0)'''
                # w_xdiffs = w * xdiffs
                w_xdiffs = np.array([gammas[n][k] * xdiffs[n] for n in range(len)])
                self.covars[k] = (1.0/Nk) * np.dot(w_xdiffs.T,xdiffs)
            loglikelyhood = 0
            '''            
            for n in range(len):
                for k in range(self.n_clusters):
                    loglikelyhood += gammas[n][k]*(np.log(self.weights[k]) + np.log(self.Gaussian(self.data[n],self.means[k],self.covars[k])))
            '''
            for k in range(self.n_clusters):
                try:
                    loglikelyhood += np.sum(gammas[:,k]*np.log(self.weights[k]) + gammas[:,k]*np.log(multivariate_normal.pdf(self.data,self.means[k],self.covars[k])))
                except:
                    continue
            i += 1

        # 屏蔽结束
    
    def fit(self, data):

        self.data = data
        self.weights = np.random.rand(self.n_clusters)
        self.weights /= np.sum(self.weights)
        dim = np.shape(self.data)[1]
        self.means = []
        #产生n_clusters个均值
        for i in range(self.n_clusters):
            mean = np.random.rand(dim)
            self.means.append(mean)
        self.covars = []
        #产生 n_clusters个协方差
        for i in range(self.n_clusters):
            cov = np.eye(dim)
            self.covars.append(cov)
        self.GMM_EM()
        for i in data:
            self.clf_[self.predict(i)].append(i)

    def predict(self,x):
        gammas = np.zeros(self.n_clusters)
        for k in range(self.n_clusters):
            try:
                prob = multivariate_normal.pdf(x,self.means[k],self.covars[k])
            except:
                continue
            gammas[k] = prob * self.weights[k]
        return np.argmax(gammas)
        
    def get_Silhouette_coefficient(self,x):
        
        a,b=0,0
        for p_data in x:
            gammas = np.zeros(self.n_clusters)
            for k in range(self.n_clusters):
                try:
                    prob = multivariate_normal.pdf(p_data,self.means[k],self.covars[k])
                except:
                    continue
                gammas[k] = prob * self.weights[k]
            smallest,smallest_index=np.inf,0
            secondsmallest,secondsmallest_index=np.inf,0
            for i,v in enumerate(gammas):
                if v<smallest:
                    secondsmallest=smallest
                    secondsmallest_index=smallest_index
                    smallest=v
                    smallest_index=i
                if secondsmallest>v>smallest:
                    secondsmallest=v
                    secondsmallest_index=i
            if len(self.clf_[smallest_index])!=0:
                for i in self.clf_[smallest_index]:
                    a+=np.linalg.norm(p_data-i)
                a/=len(self.clf_[smallest_index])
            if len(self.clf_[secondsmallest_index])!=0:
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
Accelerate_Kmean=GMM(n_clusters=3)
Accelerate_Kmean.fit(b)
predict=np.array([0]*len(b))
for i,v in enumerate(b):
    predict[i]=Accelerate_Kmean.predict(v)

print(Accelerate_Kmean.get_Silhouette_coefficient(b))    
print(Accelerate_Kmean.rand_index(label,predict))
