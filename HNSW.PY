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

class ivf :

# Parameters:
    def __init__(
        self,
        nprops,
        ) -> None:
        self.nprops=nprops

    # train
    def HNSW_train(self,data):
        HNSW= faiss.IndexHNSWFlat(70, 32)
        HNSW.add(self.ivfindex.centroids)

    #clustering_data
    def IVF_predict(self):
        xp=self.fetch_from_csv(self.data_path,self.train_batch_size+self.predict_batch_size*self.prediction_count,self.predict_batch_size)
        embeds = xp[:, 1:]
        embeds = np.array([record/np.linalg.norm(record) for record in embeds])
        self.prediction_count+=1
        assignments, _ = vq(embeds, self.centroids)
        clustering_batch=self.preprocessing(xp,assignments)
        return clustering_batch
    
    #Searching
    def _cal_score(self,vec1, vec2):
        cosine_similarity=vec1.dot(vec2.T).T / (np.linalg.norm(vec1) * np.linalg.norm(vec2))
        return cosine_similarity
    
    #nearest Centroids
    def IVF_search_combo_data(self,query):
        l=[]
        query = query/np.linalg.norm(query)
        for centroid in self.centroids:
            x=self._cal_score(query,centroid)
            x= math.sqrt(2*abs(1-x))
            l.append(x)
        nearset_centers=sorted(range(len(l)), key=lambda sub: l[sub])[:self.nprops]
        return nearset_centers
    
    #nearest idicies
    def IVF_search_small_data(self,query,clusters,top_k):
        l=[]
        query = query/np.linalg.norm(query)
        for centroid in self.centroids:
            x=self._cal_score(query,centroid)
            x= math.sqrt(2*abs(1-x))
            l.append(x)
        nearset_centers=sorted(range(len(l)), key=lambda sub: l[sub])[:self.nprops]
        nearest=[]
        for c in nearset_centers:
            for row in clusters[c]:
                x=self._cal_score(row[1], query[0])
                x= math.sqrt(2*abs(1-x))
                nearest.append({'index':row[0],'value':x})
        sorted_list = sorted(nearest, key=lambda x: x['value'])[:top_k]
        return [d['index'] for d in sorted_list]
        

    