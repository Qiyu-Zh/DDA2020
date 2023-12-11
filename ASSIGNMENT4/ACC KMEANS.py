import pandas as pd
import numpy as np
import time

def accelerated_k_means(k: int, x, centroids: list, threshold):
    # Initialization
    start_time = time.time()
    n = x.shape[0]
    lb = pd.DataFrame(0, index=list(range(n)), columns=list(range(k)))
    label = pd.Series([0] * n)
    ub = x.apply(lambda x: np.sqrt(sum((x - centroids[0]) ** 2)), axis=1)
    lb.iloc[:, 0] = ub

    centroids_distance = pd.DataFrame(index=list(range(k)), columns=list(range(k)))
    for i in range(k):
        for j in range(i, k):
            centroids_distance.iloc[i, j] = np.sqrt(sum((centroids[i] - centroids[j]) ** 2))

    for i in range(1, k):
        tmp = label.apply(lambda x: centroids_distance.iloc[x, i])
        index_1 = np.where(tmp < (2 * ub))[0]
        lb.loc[index_1, i] = x.loc[index_1, :].apply(lambda x: np.sqrt(sum((x - centroids[i]) ** 2)), axis=1)
        index_2 = ub[index_1][ub[index_1] > lb.loc[index_1, i]].index
        ub[index_2] = lb.loc[index_2, i]
        label[index_2] = i

    # Repeat until convergence
    iteration = 0
    while True:
        iteration += 1
        # Step 1
        centroids_distance = pd.DataFrame(index=list(range(k)), columns=list(range(k)))
        for i in range(k):
            for j in range(i, k):
                centroids_distance.iloc[i, j] = np.sqrt(sum((centroids[i] - centroids[j]) ** 2))
                centroids_distance.iloc[j, i] = centroids_distance.iloc[i, j]
        sc = centroids_distance.apply(lambda x: x.drop(np.where(x == 0)[0]).min(), axis=1) / 2

        # Step 2
        remaining_points = ub[ub > label.apply(lambda x: sc[x])].index

        # Step 3
        for i in remaining_points.to_list():
            for j in range(k):
                if label[i] == j:
                    continue
                elif ub[i] <= lb.iloc[i, j]:
                    continue
                elif ub[i] <= (centroids_distance.iloc[j, label[i]] / 2):
                    continue
                else:
                    # Step 3a
                    if r[i]:
                        t = np.sqrt(sum((x.iloc[i, :] - centroids[label[i]]) ** 2))
                        lb.iloc[i, label[i]] = t
                        ub[i] = t
                        r[i] = False
                    else:
                        t = ub[i]
                    # Step 3b
                    if (t > lb.iloc[i, j]) or (t > centroids_distance.iloc[j, label[i]] / 2):
                        d_x_cj = np.sqrt(sum((x.iloc[i, :] - centroids[j]) ** 2))
                        lb.iloc[i, j] = d_x_cj
                        if d_x_cj < t:
                            label[i] = j
                            ub[i] = d_x_cj

        # Step 4
        new_centroids = []
        for i in range(k):
            new_centroids.append(x.loc[label == i, :].apply(np.mean, axis=0))

        # Step 5
        d_c_mc = [np.sqrt(sum((new_centroids[i] - centroids[i]) ** 2)) for i in range(k)]
        for i in range(k):
            lb.iloc[:, i] = lb.iloc[:, i].apply(lambda x: max(0, x - d_c_mc[i]))

        # Step 6
        ub = ub + label.apply(lambda x: d_c_mc[x])
        r = pd.Series(True, index=list(range(n)))

        # Check convergence and Step 7
        delta = sum(d_c_mc)
        delta = delta / k
        if delta < threshold:
            break
        else:
            centroids = new_centroids
    end_time = time.time()
    return label, new_centroids, iteration, end_time - start_time
