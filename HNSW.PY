import math
import numpy as np
from scipy.cluster.vq import kmeans2,vq
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
from sklearn import preprocessing
import time
from dataclasses import dataclass
from typing import List
import sys
# import faiss

class ivf :

# Parameters:
    def __init__(
        self,
        nprops,
        ) -> None:
        self.nprops=nprops

    # train
    def HNSW_train(self,data):
        self.HNSW= faiss.IndexHNSWFlat(70, 32)
        self.HNSW.add(self.ivfindex.centroids)
        
     #Searching
    def _cal_score(self,vec1, vec2):
        cosine_similarity=vec1.dot(vec2.T).T / (np.linalg.norm(vec1) * np.linalg.norm(vec2))
        return cosine_similarity
    def HNSW_search(self,query):
        _, nearset_centers= self.HNSW.search(query, self.nprops)
        return nearset_centers
