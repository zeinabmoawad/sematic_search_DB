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
import faiss

class HNSW :

# Parameters:
    def __init__(
        self,
        nprops,
        ) -> None:
        self.nprops=nprops

    # train
    def HNSW_train(self,data):
        self.HNSW= faiss.IndexHNSWFlat(70, 32)
        self.HNSW.add(data)
        
    def HNSW_search(self,query):
        _, nearset_centers= self.HNSW.search(query, self.nprops)
        return nearset_centers
