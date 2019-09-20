from sklearn.cluster import KMeans
from sklearn.metrics.cluster import normalized_mutual_info_score
import numpy as np


def evaluation(X, Y, Kset):
    num = X.shape[0]
    classN = np.max(Y)+1
    kmax = np.max(Kset)
    recallK = np.zeros(len(Kset))
    #compute NMI
    kmeans = KMeans(n_clusters=classN).fit(X)
    nmi = normalized_mutual_info_score(Y, kmeans.labels_, average_method='arithmetic')
    #compute Recall@K
    sim = X.dot(X.T)
    minval = np.min(sim) - 1.
    sim -= np.diag(np.diag(sim))
    sim += np.diag(np.ones(num) * minval)
    indices = np.argsort(-sim, axis=1)[:, : kmax]
    YNN = Y[indices]
    for i in range(0, len(Kset)):
        pos = 0.
        for j in range(0, num):
            if Y[j] in YNN[j, :Kset[i]]:
                pos += 1.
        recallK[i] = pos/num
    return nmi, recallK