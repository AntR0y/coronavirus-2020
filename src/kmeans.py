import numpy as np

class KMeans():
    def __init__(self, n_clusters):
        """
        This class implements the traditional KMeans algorithm with hard assignments:

        https://en.wikipedia.org/wiki/K-means_clustering

        The KMeans algorithm has two steps:

        1. Update assignments
        2. Update the means

        While you only have to implement the fit and predict functions to pass the
        test cases, we recommend that you use an update_assignments function and an
        update_means function internally for the class.

        Use only numpy to implement this algorithm.

        Args:
            n_clusters (int): Number of clusters to cluster the given data into.

        """
        self.n_clusters = n_clusters
        self.means = None

    def fit(self, features):
        """
        Fit KMeans to the given data using `self.n_clusters` number of clusters.
        Features can have greater than 2 dimensions.

        Args:
            features (np.ndarray): array containing inputs of size
                (n_samples, n_features).
        Returns:
            None (saves model - means - internally)
        """
        # initialize means
        self.means = features[np.random.randint(features.shape[0], size=self.n_clusters), :]
        old_means = np.zeros((self.n_clusters, features.shape[1]))
        labels = np.zeros(features.shape[0])

        while not np.array_equal(self.means, old_means):
            old_means = self.means.copy()
            self.update_assignments(features, labels)
            self.update_means(features, labels)

    def update_assignments(self, features, labels):
        for row in range(features.shape[0]):
            dists = np.linalg.norm(self.means - features[row,:], axis=1)
            labels[row] = np.argmin(dists)

    def update_means(self, features, labels):
        for cluster in range(self.means.shape[0]):
            bool_ind = labels == cluster
            self.means[cluster] = np.mean(features[bool_ind,:], axis=0)

    def predict(self, features):
        """
        Given features, an np.ndarray of size (n_samples, n_features), predict cluster
        membership labels.

        Args:
            features (np.ndarray): array containing inputs of size
                (n_samples, n_features).
        Returns:
            predictions (np.ndarray): predicted cluster membership for each features,
                of size (n_samples,). Each element of the array is the index of the
                cluster the sample belongs to.
        """
        predictions = np.zeros(features.shape[0])
        self.update_assignments(features, predictions)
        return predictions
