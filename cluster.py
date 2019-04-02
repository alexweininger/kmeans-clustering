from sklearn.cluster import KMeans
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import scipy as sc
import random, math
import numpy_indexed as npi
import kmeans
def Euclidean_distance(feat_one, feat_two):

    squared_distance = 0

    #Assuming correct input to the function where the lengths of two features are the same

    for i in range(len(feat_one)):
        squared_distance += (feat_one[i] - feat_two[i])**2
    ed = math.sqrt(squared_distance)

    return ed

def plotData(X, centroids, labels=None):
	colors = ["red", "green", "blue", "teal", "black", "orange", "yellow", "purple"]
	if not (labels is None):
		for x in range(len(X)):
			plt.scatter(X[x][0],X[x][1], label='True Position', c=colors[labels[x]])
	else:
		plt.scatter(X[:,0],X[:,1], label='True Position', c='blue')
	plt.scatter(centroids[:,0],centroids[:,1], label='True Position', c='red')
	avg = np.average(X[:,0])
	avgy = np.average(X[:,1])
	plt.scatter(avg, avgy, c='green')

	plt.show()

def knn(clusters, data, plot=False):
	est = KMeans(n_clusters=clusters).fit(data)
	print(est.cluster_centers_)
	print(est.labels_)
	entropy(est.labels_, clusters)
	if plot:
		plotData(data, est.cluster_centers_)

def entropy(labels, clusters):
	s = pd.read_csv('./synthetic/unbalance-gt.pa', header=None, index_col=False)[0]
	totals = np.zeros((clusters, clusters))

	for label, true in zip(labels, s):
		totals[label - 1][true - 1] += 1

	pk = [[val / sum(totals[cluster]) for val in totals[cluster]] for cluster in range(clusters)]
	e_vals = sc.stats.entropy(pk)

	counts = np.zeros((8))
	for label in labels:
		counts[label] += 1

	pc = [count / sum(counts) for count in counts]

	mean_entropy = sum([p * e for p, e in zip(pc, e_vals)])
	print(f'mean entropy {mean_entropy}')

datas = pd.read_csv('./synthetic/unbalance.txt', sep=' ')
# datas = pd.read_csv('./MopsiLocationsUntil2012-Finland.txt')
# datas = pd.read_csv('./MopsiLocations2012-Joensuu.txt')
print(np.array(datas))
knn(8, np.array(datas), True)



class K_Means:
	def __init__(self, k =3, tolerance = 0.0001, max_iterations = 300):
		self.k = k
		self.tolerance = tolerance
		self.max_iterations = max_iterations

	def fit(self, data):
		self.labels = []
		#initialize the centroids, the first 'k' elements in the dataset will be our initial centroids
		self.centroids = np.array([data[random.randrange(len(data))] for r in range(self.k)])
		self.centroids = np.array(data[-self.k:])

		#begin iterations
		for i in range(self.max_iterations):
			# classes hold the instances associated with each cluster/centroid
			self.classes = [[] for i in range(self.k)]
			self.labels = []
			#find the distance between the point and cluster; choose the nearest centroid
			for features in data:
				distances = [Euclidean_distance(features, self.centroids[z]) for z in range(self.k)]
				classification = distances.index(min(distances))
				self.labels.append(classification)
				self.classes[classification].append(features)


			previous = self.centroids

			#average the cluster datapoints to re-calculate the centroids
			for classification in range(self.k):
				self.centroids[classification] = np.average(self.classes[classification], axis = 0)

			# isOptimal = True

			# for centroid in range(self.k):

			# 	original_centroid = previous[centroid]
			# 	curr = self.centroids[centroid]

			# 	if np.sum((curr - original_centroid)/original_centroid * 100.0) > self.tolerance:
			# 		isOptimal = False

			# #break out of the main loop if the results are optimal, ie. the centroids don't change their positions much(more than our tolerance)
			# if isOptimal:
			# 	break

	def pred(self, data):
		distances = [np.linalg.norm(data - self.centroids[centroid]) for centroid in self.centroids]
		classification = distances.index(min(distances))
		return classification

km = K_Means(k=8, tolerance=0.0001)
km.fit(np.array(datas))
print(km.centroids)
plotData(np.array(datas), km.centroids)

c, clusters = kmeans.kmeans(np.mat(datas), 8)
plotData(np.array(datas), c)
