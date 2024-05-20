from benchmark.algorithms.base.module import BaseDBSCAN

import dbscan_srr as dbscan
import numpy as np

class SRRDBSCAN(BaseDBSCAN):
    def __init__(self, delta, memory, threads, shrinkage):
        self.delta = delta
        self.memory = memory
        self.threads = threads
        self.shrinkage = shrinkage # only look at shrinkage * 100% of repetitions

    def cluster(self, X: np.array, epsilon: float, minPts:int):  
        self.clustering = dbscan.SRR().fit_predict(X, self.delta, self.memory, True, "test", self.threads, -1, self.shrinkage, epsilon, minPts)

    def retrieve_labels(self):
        self.clustering = np.array(self.clustering)
        return self.clustering, self.clustering[self.clustering > -1], [] 

    
    def __str__(self):
        return f"SRRDBSCAN(delta={self.delta}, memory={self.memory}, threads={self.threads}, shrinkage={self.shrinkage})"

    def __repr__(self):
        return f"{self.delta}_{self.memory}_{self.threads}, {self.shrinkage}"
