class BaseDBSCAN:
    def __init__(self):
        raise NotImplementedError()

    def cluster(self, X, minPts: int, epsilon: float):
        """Clusters the data."""
        raise NotImplementedError()

    def retrieve_labels(self):
        """Returns array of length len(X), with individual labels of clusters. -1 is point is classified as noise."""
        raise NotImplementedError()