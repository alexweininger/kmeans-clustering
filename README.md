Usage instructions:

The only file needed to run is kmeans.py. It relies upon ./synthetic and the other data sets to read from and run.

to run: py kmeans.py

Put the synthetic data in a directory named synthetic in the project's root. And the rest of the data sets can be in the root also.

Architecture:

The bulk of the work takes place in the kmeans function in the kmeans.py file. This is where the centroids are computed by calculating the average of the points
of that cluster. To detirmine what points are in what cluster, the euclidian distance function is used to put the points in the cluster that has the centroid
closest to the point.
