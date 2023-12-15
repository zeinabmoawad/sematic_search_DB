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
import pickle
import platform
import struct
import os


def save_file(file_path, file_save):
    with open(file_path, 'wb') as file:
        pickle.dump(file_save, file)


def load_centroids(file_path):
    # check if file exists
    if not os.path.exists(file_path):
        print("File does not exist")
        exit()
    with open(file_path, 'rb') as file:
        file_loaded = pickle.load(file)
    return file_loaded

class ivf :

# Parameters:
    def __init__(
        self,
        data_path,
        train_batch_size,
        predict_batch_size,
        nprops,
        iter,
        centroids_num,
        folder_path,
        load = False

        ) -> None:
        self.train_batch_size=train_batch_size
        self.predict_batch_size=predict_batch_size
        self.data_path=data_path
        self.prediction_count=0
        self.nprops=nprops
        self.iter=iter
        self.centroids_num=centroids_num
        self.folder_path=folder_path
        self.load = load
        if self.load:
            self.centroids = load_centroids(self.folder_path+"centroids.pkl")

    def fetch_from_binary(self,file_path,line,size):
      print(line, size)
      with open(file_path, "rb") as binary_file:
        binary_file.seek(line*564)
        # Read an integer (4 bytes) from the current position
        rows=[]
        for i in range(size):
          packed_integer = binary_file.read(4)
          integer_value = struct.unpack('i', packed_integer)[0]
          # Move the cursor to a specific position (e.g., 4 bytes from the beginning)
          # Read a float (8 bytes) from the current position
          packed_float = binary_file.read(8*70)
          float_values = []
          float_values = [struct.unpack('d', packed_float[i:i+8])[0] for i in range(0, len(packed_float), 8)]
          float_values.insert(0,integer_value)
          rows.append(float_values)
        return np.array(rows)

    #Assign every batch
    def preprocessing(self,xp,assignments):
        clustering_batch = [[] for _ in range(len(self.centroids))]
        for i, k in enumerate(assignments):
            clustering_batch[k].append((xp[i][0],xp[i][1:]))  # the nth vector gets added to the kth cluster...
        return clustering_batch
    # train
    def IVF_train(self):
        if not self.load:
            xp=self.fetch_from_binary(self.data_path,0,self.train_batch_size)
            print((xp)[:,0])
            embeds = xp[:, 1:]
            embeds = np.array([record/np.linalg.norm(record) for record in embeds])
            (centroids, assignments) = kmeans2(embeds, self.centroids_num, self.iter,minit='points')
            self.centroids=centroids
            save_file(self.folder_path+"centroids.pkl", self.centroids)
            clustering_batch=self.preprocessing(xp,assignments)
            return clustering_batch

    #clustering_data
    def IVF_predict(self):
        xp=self.fetch_from_binary(self.data_path,self.train_batch_size+self.predict_batch_size*self.prediction_count,self.predict_batch_size)
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
        copy_query = query.copy()
        copy_query = copy_query/np.linalg.norm(copy_query)
        for centroid in self.centroids:
            x=self._cal_score(copy_query,centroid)
            x= math.sqrt(2*abs(1-x))
            l.append(x)
        nearset_centers=sorted(range(len(l)), key=lambda sub: l[sub])[:self.nprops]
        return nearset_centers
    
    #nearest idicies
    def IVF_search_small_data(self,query,top_k):
        l=[]
        query = query/np.linalg.norm(query)
        for centroid in self.centroids:
            x=self._cal_score(query,centroid)
            x= math.sqrt(2*abs(1-x))
            l.append(x)
        nearset_centers=sorted(range(len(l)), key=lambda sub: l[sub])[:self.nprops]
        nearest=[]
        for c in nearset_centers:
                clusters = self.load_from_binary_file(self.folder_path+"ivf_cluster_"+str(c)+".bin")
                for row in clusters:
                    x=self._cal_score(np.array(row[1]), query[0])
                    x= math.sqrt(2*abs(1-x))
                    nearest.append({'index':row[0],'value':x})
        sorted_list = sorted(nearest, key=lambda x: x['value'])[:top_k]
        return [d['index'] for d in sorted_list]
    
    def add_clusters(self,cluster:np.ndarray= None) -> None:
        """Add vectors to the database (their encoded versions).

        Parameters
        ----------
        X
            Array of shape (n_codes, d) of dtype np.float32.
        """
        if cluster is not None:
            for i in range(len(cluster)):
                for item in cluster[i]:
                    with open(self.folder_path+"ivf_cluster_"+str(i)+".bin", 'ab') as file:
                        file.write(struct.pack('i',int(item[0])))  
                        file.write(struct.pack(f'{len(item[1])}d', *item[1]))
            
    def load_from_binary_file(self,file_path):
        """
        Load data from a text file.

        Parameters:
        - file_path: Path to the file.

        Returns:
        - List of tuples loaded from the file.
        """
        loaded_data = []
        with open(file_path, "rb") as binary_file:
            # Loop through the file until the end
            while True:
                # Read an integer (4 bytes)
                packed_integer = binary_file.read(4)
                if not packed_integer:
                    # Break the loop if there is no more data to read
                    break
                integer_value = struct.unpack('i', packed_integer)[0]

                # Read a list of floats (8 bytes each)
                packed_floats = binary_file.read(8 *70)  # Assuming a list of 3 floats per row
                float_values = [struct.unpack('d', packed_floats[i:i+8])[0] for i in range(0, len(packed_floats), 8)]

                # Append the result as a tuple (integer, list of floats)
                loaded_data.append((integer_value, float_values))
            return loaded_data
        
