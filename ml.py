import numpy as np
class KMeans:
    '''
    Basic KMeans clustering using numpy
    '''
    def __init__(self,n = 2,max_iter = 10, atol = 0.01):
        '''
        Initialize the kmeans class.
        '''
        self.n = n
        self.max_iter = max_iter
        self.atol = atol
        self.clusters = None
        self.centroids = []
        self.inertia = 0
    def fit(self,X):
        """
        This method implements the k-means clustering algorithm to fit the model to the data.

        Parameters:
        X (numpy.ndarray): A 2D array where each row is a separate data point and each column is a feature.

        Returns:
        None. The method updates the model's centroids and clusters in-place.

        Raises:
        ValueError: If the number of data points in X is less than or equal to twice the number of clusters.

        Note:
        The method uses Euclidean distance to assign each data point to the cluster whose centroid is closest.
        The centroids are then updated as the mean of the data points in each cluster.
        This process is repeated until a maximum number of iterations is reached.
        """
        if len(X) <= self.n * 2:
            raise ValueError('Not enough data points for the number of clusters')
            #return 'Need more data points'
        rng = np.random.default_rng(0)
        #selects random points for initial clusters
        while len(np.unique(self.centroids)) < self.n:
            self.centroids = X[rng.integers(0,len(X), self.n)]
        max_iter = 0
        # Core loop for updating the clusters and centroids
        while max_iter < self.max_iter:
            # Create empty clusters 
            self.clusters = [[] for i in range(self.n)]
            for feature in X:
                min_dist = np.inf
                for i,centroid in enumerate(self.centroids):
                    distance = np.sqrt(np.sum(np.power(feature - centroid,2)))
                    if distance < min_dist:
                        cluster_index = i
                        min_dist = distance
                self.clusters[cluster_index].append(feature)        
            self.centroids = [np.array(points).mean(axis = 0) for points in self.clusters]
            max_iter += 1
        self.centroids = np.array(self.centroids)

            
    def calculate_inertia(self):
        '''
        not used !
        '''
        total = 0
        i = 0
        for centroid in self.centroids:
            for cluster in self.clusters:
                total += np.sum(np.power(centroid - cluster,2))
                i += 1
        if i > 0:
            self.inertia = total / i
        else:
            self.inertia = 0
        return self.inertia

    def predict(self,X):
        """
        This method predicts the cluster index for each data point in X based on the model's centroids.

        Parameters:
        X (numpy.ndarray): A 2D array where each row is a separate data point and each column is a feature.

        Returns:
        preds (list): A list of predicted cluster indices for each data point in X.

        Note:
        The method uses Euclidean distance to assign each data point to the cluster whose centroid is closest.
        """
        preds = []
        for feature in X:
            min_dist = np.inf
            for i,centroid in enumerate(self.centroids):
                distance = np.sqrt(np.sum(np.power(feature - centroid,2)))
                if distance < min_dist:
                    min_dist = distance
                    cluster_index = i
            preds.append(cluster_index)
        return preds


if __name__ == "__main__":


    kmeans = KMeans(n = 4,max_iter = 15 )
    #kmeans.fit(X)
  