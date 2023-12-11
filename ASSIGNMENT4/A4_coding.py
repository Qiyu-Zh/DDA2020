#%%

import numpy as np
import matplotlib.pyplot as plt
import random as rand
import math
from scipy.stats import multivariate_normal
import time

#%%

#initialization
def loadData():
    f = open("seeds_dataset.txt", 'r')
    dataset = np.empty([210,7])
    class_index = np.empty([210,1])
    for i in range(0,210):
        elements = f.readline().rstrip().split('\t')   
        index = 0                                        
        for j in range(len(elements)):
            if elements[j] != '':
                dataset[i][index] = elements[j]        
                index += 1 
                if index == 7:
                    break
        class_index[i] = elements[len(elements)-1]       

    return dataset, class_index

def init_rand_value(dataset):
    rand.seed()
    max_array = np.amax(dataset,axis=0)                 
    min_array = np.amin(dataset,axis=0)
    class_index = np.empty(210)
    center = np.empty([3,7])
    for i in range(0,210):
        class_index[i] = rand.randrange(1,4)               #generate random initial state for each element

    for i in range(0,7):                                   #generate three random center of cluster
        center[0][i] = rand.uniform(min_array[i], max_array[i])
        center[1][i] = rand.uniform(min_array[i], max_array[i])
        center[2][i] = rand.uniform(min_array[i], max_array[i])

    return class_index, center

#%%

#K-means
def New_c_i(data, center):                     #decide the class of a data
    
    d1 = np.linalg.norm(data - center[0])      #get the distance
    d2 = np.linalg.norm(data - center[1])
    d3 = np.linalg.norm(data - center[2])

    i = 1                                      #find the smallest one
    d_min = d1

    if (d2 < d_min):
        d_min = d2
        i = 2

    if (d3 < d_min):
        d_min = d3
        i = 3

    return i

def New_center_k(dataset, class_index):                     # calculate the new center of datas
    n1 = 0
    n2 = 0
    n3 = 0
    center_new = np.empty([3,7])


    for i in range(0,210):
        if class_index[i] == 1:
            center_new[0] += dataset[i]
            n1 += 1

        elif class_index[i] == 2:
            center_new[1] += dataset[i]
            n2 += 1

        else:
            center_new[2] += dataset[i]
            n3 += 1

            

    center_new[0] /= n1
    center_new[1] /= n2
    center_new[2] /= n3

    return center_new


def k_means(dataset, class_index_pre, center_pre):
    class_index = np.copy(class_index_pre)
    last_class = np.zeros(np.size(class_index))

    center = np.copy(center_pre)
    last_center = np.zeros(np.size(center))

    iteration = 0
    start_time = time.time()

    while np.linalg.norm(class_index - last_class) > 0 and \
            (np.linalg.norm(center[0]-last_center[0])+
             np.linalg.norm(center[1]-last_center[1])+
             np.linalg.norm(center[2]-last_center[2]))>0.01:      
       
        last_class = np.copy(class_index)                         
        for i in range(0,210):
            class_index[i] = New_c_i(dataset[i],center) 
        

        last_center = np.copy(center)                              
        center = New_center_k(dataset,class_index)
        iteration += 1

    end_time = time.time()

    return class_index, last_center, iteration, end_time-start_time


#%%

# Accelerated K-means
def acc_K_means(dataset, class_index_pre, center_pre):
    class_index = np.copy(class_index_pre)
    center = np.ones(np.size(center_pre))

    iteration = 0

    start_time = time.time()

    l_bound = np.zeros([210,3])
    u_bound = np.zeros(210)

    center_d = np.zeros([3,3])

    center_d[0][1] = np.linalg.norm(center[0]-center[1])
    center_d[1][0] = np.linalg.norm(center[0]-center[1])

    center_d[0][2] = np.linalg.norm(center[0]-center[2])
    center_d[2][0] = np.linalg.norm(center[0]-center[2])

    center_d[2][1] = np.linalg.norm(center[1]-center[2])
    center_d[1][2] = np.linalg.norm(center[1]-center[2])


    for i in range(0,210):
        l_bound[i][0] = np.linalg.norm(dataset[i]-center[0])
        u_bound[i] = l_bound[i][0]
        
        for j in range(1,3):
            # Triangle Inequality
            if 0.5 * center_d[int(class_index[i]-1)][j] < l_bound[i][int(class_index[i]-1)]:
                d = np.linalg.norm(dataset[i] - center[j])
                l_bound[i][j] = d
                if u_bound[i] > d:
                    u_bound[i] = d
                if l_bound[i][int(class_index[i] - 1)] > d:
                    class_index[i] = j + 1


    sc = np.zeros(3)
    rx = np.zeros(210)
    last_center = np.zeros(np.size(center))

    while np.linalg.norm(last_center[0]-center[0])\
            +np.linalg.norm(last_center[1]-center[1])\
            + np.linalg.norm(last_center[2]-center[2]) > 0.01:

        iteration += 1


        #step1
        # compute d(c,c') for all centers c and c'
        center_d = np.zeros([3,3])
        center_d[0][1] = center_d[1][0] = np.linalg.norm(center[0]-center[1])
        center_d[0][2] = center_d[2][0] = np.linalg.norm(center[0]-center[2])
        center_d[2][1] = center_d[1][2] = np.linalg.norm(center[1]-center[2])
        sc[0] = 0.5 * min(center_d[0][1], center_d[0][2])
        sc[1] = 0.5 * min(center_d[1][0], center_d[1][2])
        sc[2] = 0.5 * min(center_d[2][1], center_d[2][0])

        #step 2
        mark = np.zeros(210)
        for i in range(210):
            c_index = int(class_index[i]-1)
            if u_bound[i]<=sc[c_index]:
                mark[i] = 1

        #step 3
        for j in range(3):
            for i in range(210):
                if mark[i] == 1:
                    continue
                elif j == class_index[i] - 1 :
                    continue
                elif u_bound[i]>l_bound[i][j] and \
                        u_bound[i]>0.5 * np.linalg.norm(center[int(class_index[i]-1)]-center[j]):
                    if rx[i] == 1:
                        d = np.linalg.norm(dataset[i]-center[int(class_index[i]-1)])
                        rx[i] = 0
                    else:
                        d = u_bound[i]
                    if d > l_bound[i][j] or d > 0.5 * center_d[int(class_index[i]-1)][j]:
                        d_x_c = np.linalg.norm(dataset[i]-center[j])
                        if d_x_c<d:
                            class_index[i] = j + 1


        # step 4
        new_center = np.zeros([3,7])
        temp = np.zeros(3)

        for i in range(210):
            center_index = int(class_index[i] - 1)
            new_center[center_index] += dataset[i]
            temp[center_index] += 1

        for i in range(3):
            if temp[i] != 0:
                new_center[i] /= temp[i]

        # step 5
        for i in range(210):
            for j in range(3):
                d = np.linalg.norm(center[j] - new_center[j])
                l_bound[i][j] = max(l_bound[i][j]-d,0)

        # step 6
        for i in range(210):
            center_index = int(class_index[i] - 1)
            d = np.linalg.norm(new_center[center_index] - center[center_index])
            u_bound[i] = u_bound[i] + d
            rx[i] = 1

        # step 7
        last_center = np.copy(center)
        center = np.copy(new_center)


    end_time = time.time()

    return class_index, center, iteration, end_time-start_time
 
#%%

def processResult(r):
    u = np.ones(210)
    for i in range(210):
        max = r[i][0]
        idx = 1
        if r[i][1]>max:
            idx = 2
            max = r[i][1]
        elif r[i][2]>max:
            idx = 3
            max = r[i][2]
        u[i] = idx
    return u

#%%

# Gaussian Mixture Model
def phi(dataset, mu, cov):
    norm = multivariate_normal(mean = mu, cov = cov)
    return norm.pdf(dataset)

# E step
def E_step(dataset, mu, cov, pik):
    N = dataset.shape[0]
    K = pik.shape[0]
    gamma = np.mat(np.zeros((N, K)))
    prob = np.zeros((N, K))
    for k in range(K):
        prob[:, k] = phi(dataset, mu[k], cov[k])
    prob = np.mat(prob)

    for k in range(K):
        gamma[:, k] = pik[k] * prob[:, k]

    for i in range(N):
        gamma[i, :] /= np.sum(gamma[i, :])

    return gamma

# M step
def M_step(dataset, gamma):
    N, D = dataset.shape
    K = gamma.shape[1]
    cov = []
    mu = np.zeros((K, D))
    weight_pi = np.zeros(K)
    for k in range(K):
        Nk = np.sum(gamma[:, k])
        mu[k, :] = np.sum(np.multiply(dataset, gamma[:, k]), axis=0) / Nk
        cov_k = (dataset - mu[k]).T * np.multiply((dataset - mu[k]), gamma[:, k]) / Nk
        cov.append(cov_k)
        weight_pi[k] = Nk / N

    cov = np.array(cov)
    return mu, cov, weight_pi

# Normalizing the data
def normalize_data(Y):
    for i in range(Y.shape[1]):
        max_ = Y[:, i].max()
        min_ = Y[:, i].min()
        Y[:, i] = (Y[:, i] - min_) / (max_ - min_)

    return Y

def init_params(shape, K):
    N, D = shape
    mu = np.random.rand(K, D)
    cov = np.array([np.eye(D)] * K)
    alpha = np.array([1.0 / K] * K)
    return mu, cov, alpha
#%%

# gmm em 
# K: number of cluster
def GMM_EM(dataset, K):
    dataset = normalize_data(dataset)
    mu, cov, alpha = init_params(dataset.shape, K)
    iteration = 0
    start_time = time.time()
    while 1:
        gamma = E_step(dataset, mu, cov, alpha)
        mu_new, cov_new, alpha_new = M_step(dataset, gamma)
        iteration += 1
        if np.linalg.norm(mu-mu_new)<=0.01 and \
                np.linalg.norm(cov-cov_new)<=0.01 and \
                np.linalg.norm(alpha-alpha_new)<=0.01:
            break
        mu = mu_new
        cov = cov_new
        alpha = alpha_new

    end_time = time.time()
    duration = end_time - start_time
    return mu, cov, alpha, iteration, duration

def cluster_UsingGMMEM(dataset):
    mu, cov, alpha, itter, duration = GMM_EM(dataset, 3)
    N = dataset.shape[0]
    gamma = E_step(dataset, mu, cov, alpha)
    category = gamma.argmax(axis=1).flatten().tolist()[0]
    category1 = np.zeros(len(category))
    for i in range(np.size(category)):
        category1[i] = category[i]+1

    return category1, itter, duration
#%%

# Get the label
def getCluster(c):
    label = np.zeros(3)
    for i in range(3):
        count = [0] * 3
        for j in range(70*i, 70*(i+1)):
            if c[j] == 1:
                count[0] += 1
            elif c[j] == 2:
                count[1] += 1
            else:
                count[2] += 1
        label[i] = count.index(max(count)) + 1

    return label

def PURITY(c):
    label = getCluster(c)
    pure = np.zeros(3)
    for i in range(3):
        for j in range(70*i, 70*(i+1)):
            if c[j] == label[i]:
                pure[i] += 1

    return np.sum(pure/210)

# functions for Rand Index

# pairs
def Pairs(c, cluster):
    label = getCluster(c)
    # get size of each c
    size = np.zeros(3)
    for i in range(np.size(cluster)):
        if c[i] == label[0]:
            size[0] += 1
        elif c[i] == label[1]:
            size[1] += 1
        else:
            size[2] += 1

    pairs = np.zeros(3)
    for i in range(3):
        pairs[i] = np.math.factorial(size[i])/(2*np.math.factorial(size[i]-2))

    return np.sum(pairs)

# TP
def TP(c, cluster):
    label = getCluster(c)
    size = np.zeros(3)
    for i in range(np.size(cluster)):
        if c[i] == label[0]:
            size[0] += 1
        elif c[i] == label[1]:
            size[1] += 1
        else:
            size[2] += 1
    
    pure = np.zeros(3)
    for i in range(3):
        for j in range(70*i, 70*(i+1)):
            if c[j] == label[i]:
                pure[i] += 1
    
    tp = np.zeros(3)
    for i in range(3):
        tp[i] = np.math.factorial(pure[i])/(2*np.math.factorial(pure[i]-2))
    
    return np.sum(tp)

# FP
# TP + FP = # of pairs from the same cluster
def FP(c, cluster):
    pairs = Pairs(c, cluster)
    tp = TP(c, cluster)
    return pairs - tp

# FN
# TP + FN = # of similar pairs
def FN(c, cluster):
    tp = TP(c, cluster)
    pairs = np.zeros(3)
    for i in range(3):
        pairs[i] = np.math.factorial(70)/(2*np.math.factorial(70-2))
    
    return np.sum(pairs)-tp

# TN
# TN + FP = # of dissimilar paris
def TN(c, cluster):
    fp = FP(c, cluster)
    pairs = np.zeros(3)
    for i in range(3):
        pairs[i] = np.math.factorial(70)/(2*np.math.factorial(70-2))
    
    dpair = np.math.factorial(210)/(2*np.math.factorial(210-2)) - sum(pairs)
    return dpair - fp

# Rand Index
def RI(c, cluster):
    fp = FP(c, cluster)
    fn = FN(c, cluster)
    tp = TP(c, cluster)
    tn = TN(c, cluster)
    ri = (tp+tn)/(tp+fp+fn+tn)
    return ri

# functions for Normalized Mutual Information
def mutual_info(c, cluster):
    label = getCluster(c)
    N = np.size(cluster)
    pure = np.zeros(3)
    for i in range(3):
        for j in range(70*i, 70*(i+1)):
            if c[j] == label[i]:
                pure[i] += 1

    size = np.zeros(3)
    for i in range(N):
        if c[i] == label[0]:
            size[0] += 1
        elif c[i] == label[1]:
            size[1] += 1
        else:
            size[2] += 1

    mutual_info = np.zeros(3)
    for i in range(3):
        mutual_info[i] = pure[i]/N*np.log((N*pure[i])/(size[i]*70))

    return np.sum(mutual_info)

# entropy
from sklearn.metrics import silhouette_score as S_coef
def entropy(c, cluster):
    N = np.size(cluster)
    label = getCluster(c)
    size = np.zeros(3)
    for i in range(N):
        if c[i] == label[0]:
            size[0] += 1
        elif c[i] == label[1]:
            size[1] += 1
        else:
            size[2] += 1

    entropy = np.zeros(3)
    for i in range(3):
        entropy[i] = -size[i]/N*np.log(size[i]/N)

    return np.sum(entropy)

# NMI
def NMI(c, cluster):
    MIF = mutual_info(c, cluster)
    E1 = entropy(cluster, cluster)
    E2 = entropy(c, cluster)
    NMI = MIF/(E1+E2)*2
    return NMI


#%%

dataset, cluster = loadData()
c, u = init_rand_value(dataset)


# K means
c_out1, u_out1, itter1, duration1 = k_means(dataset,c,u)
ri1 = RI(c_out1, cluster)
nmi1 = NMI(c_out1, cluster)
purity1 = PURITY(c_out1)
score1 = S_coef(dataset,c_out1)
print("K means:")
print("iteration:", itter1)
print("Time:", duration1)
print("Purity:", purity1)
print("RI:", ri1)
print("NMI:", nmi1)
print("S_SCORE:", score1)


# gmm em
c_out2, itter2, duration2 = cluster_UsingGMMEM(dataset)
ri2 = RI(c_out2, cluster)
nmi2 = NMI(c_out2, cluster)
purity2 = PURITY(c_out2)
score2 = S_coef(dataset, c_out2)
print("GMM EM:")
print("iteration:", itter2)
print("Time:", duration2)
print("Purity:", purity2)
print("RI:", ri2)
print("NMI:", nmi2)
print("S_SCORE:", score2)


# acc_K means
c_out3,u_out3, itter3, duration3 = acc_K_means(dataset,c,u)
#ri3 = RI(c_out3, cluster)
nmi3 = NMI(c_out3, cluster)
purity3 = PURITY(c_out3)
#score3 = S_coef(dataset, c_out3)
print("Acc k means:")
print("iteration:", itter3)
print("Time:", duration3)
print("Purity:", purity3)
#print("RI:", ri3)
print("NMI:", nmi3)
#print("S_SCORE:", score3)


#%%

#k means
pure_list1 = np.zeros(100)
for i in range(100):
    c, u = init_rand_value(dataset)
    c_out1, u_out1, iteration1, duration1 = k_means(dataset, c, u)
    pure_list1[i] = PURITY(c_out1)
x = list(range(100))
plt.figure()
plt.plot(x, pure_list1)
plt.show()

#%%

#GMM EM
pure_list2 = np.zeros(100)
for i in range(100):
    c, u = init_rand_value(dataset)
    c_out2, itter2, duration2 = cluster_UsingGMMEM(dataset)
    pure_list2[i] = PURITY(c_out2)
x = list(range(100))
plt.figure()
plt.plot(x, pure_list2)
plt.show()

#%%

#acc k means
pure_list3 = np.zeros(100)
for i in range(100):
    c, u = init_rand_value(dataset)
    c_out3,u_out3, itter3, duration3 = acc_K_means(dataset,c,u)
    pure_list3[i] = PURITY(c_out3)
x = list(range(100))
plt.figure()
plt.plot(x, pure_list3)
plt.show()