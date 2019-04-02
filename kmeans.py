import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import math, random
import scipy.stats as sc

# gets the euclidian distance between two coordinates a and b
def euclidian_dist(a, b):
    dist = 0.0
    for x in range(len(a)):
        dist += (a[x] - b[x])**2
    return math.sqrt(dist)

def entropy(labels, clusters):
	s = pd.read_csv('./synthetic/unbalance-gt.pa', header=None, index_col=False)[0]
	totals = np.zeros((clusters, clusters))

	for label, true in zip(labels, s):
		label = int(label)
		totals[label - 1][true - 1] += 1

	pk = [[val / sum(totals[cluster]) for val in totals[cluster]] for cluster in range(clusters)]
	e_vals = sc.entropy(pk)

	counts = np.zeros((8))
	for label in labels:
		label = int(label)
		counts[label] += 1

	pc = [count / sum(counts) for count in counts]

	mean_entropy = sum([p * e for p, e in zip(pc, e_vals)])
	print(f'mean entropy {mean_entropy}')

def kmeans(data, k, max_iters=50):
    centroids = []
    labels = None
    for x in range(k):
        centroids += [random.choice(data)]

    for j in range(max_iters):

        distances = np.zeros(len(data))
        labels = np.zeros(len(data))

        for i in range(len(data)):
            centroid_distances = np.zeros((k))
            for x in range(k):
                centroid_distances[x] = euclidian_dist(
                    data[i], centroids[x])
            distance_index = centroid_distances.tolist().index(min(centroid_distances))
            distances[i] = min(centroid_distances)
            labels[i] = distance_index

        #
        total = np.zeros((k, k))
        features = np.zeros((k))
        for x in range(len(data)):
            index = int(labels[x])
            total[index][0] += data[x][0]
            total[index][1] += data[x][1]
            features[index] += 1

        # calculate the average of the features in the cluster to calculate the value of the centroids
        for x in range(len(features)):
            if min(total[x]) == 0:
                centroids[x][0] = (total[x][0] / features[x])
                centroids[x][1] = (total[x][1] / features[x])
            else:
                centroids[x] = random.choice(data)

        # this code makes sure the clusters are not too close together
        if k == 8:
            centroid_Euclid = np.zeros((8, 8))
            for x in range(len(centroids)):
                for i in range(len(centroids)):
                    # in order to not get 0 in the array
                    if x == i:
                        centroid_Euclid[x][i] = 999999
                    else:
                        centroid_Euclid[x][i] = euclidian_dist(
                            centroids[x], centroids[i])

            for x in range(len(centroid_Euclid)):
                for i in range(len(centroid_Euclid)):
                    if centroid_Euclid[x][i] <= 40000:
                        centroids[x] = random.choice(data)
    return np.array(centroids), labels


def plot_clusters(data, centroids):
    plt.scatter(data[:, 0], data[:, 1], c='blue')
    plt.scatter(centroids[:, 0], centroids[:, 1], c='red')
    plt.show()

datas = pd.read_csv('./synthetic/unbalance.txt', sep=' ')
# datas = pd.read_csv('./MopsiLocationsUntil2012-Finland.txt')
# datas = pd.read_csv('./MopsiLocations2012-Joensuu.txt')

datas = np.array(datas)

centroids, labels = np.array(kmeans(datas, 8))

entropy(np.array(labels), 8) # works only for synthetic data because there is no ground truth for the other data sets

np.set_printoptions(precision=3)
print('Centroids: \n', centroids)
plot_clusters(datas, centroids)
