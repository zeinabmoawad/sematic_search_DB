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
    # if not os.path.exists(file_path):
    #     os.makedirs(os.path.dirname(file_path), exist_ok=True)
    with open(file_path, 'wb') as file:
        pickle.dump(file_save, file)


def load(file_path):
    with open(file_path, 'r') as file:
        file_loaded = pickle.load(file)
    return file_loaded

class ivf :

# Parameters:
    def __init__(
        self,
        data_path,
        train_batch_size,
        predict_batch_size,
        # k,
        nprops,
        iter,
        centroids_num,
        centroid_path,
        load = False
        ) -> None:
        self.train_batch_size=train_batch_size
        self.predict_batch_size=predict_batch_size
        self.data_path=data_path
        self.prediction_count=0
        self.nprops=nprops
        self.iter=iter
        self.centroids_num=centroids_num
        self.centroid_path = centroid_path
        self.load = load
        if self.load:
            self.centroids = load(self.centroid_path)

    #Fetching file
    # def fetch_from_csv(self,file_path,line_number,size):
    #     with open(file_path, 'r') as fp:
    #         x=fp.readlines()[line_number-1:line_number+size-1]
    #         x=[np.fromstring(row, dtype=float, sep=',')for row in x]
    #         return np.array(x)

    # def fetch_from_csv(self,file_path,line_number,size):
    #     row_size = 639 #size of each row in bytes
    #     if platform.system() == "Linux":
    #         row_size = 638
    #     byte_offset = self.calculate_byte_offset(line_number, row_size)
    #     specific_rows=[]
    #     with open(file_path, 'r', encoding='utf-8') as csv_file:
    #         csv_file.seek(byte_offset)
    #         specific_row = csv_file.readline().strip()
    #         specific_row = np.fromstring(specific_row, dtype=float, sep=',')
    #         specific_rows.append(specific_row)
    #         for i in range(size-1):
    #             specific_row = csv_file.readline().strip()
    #             specific_row = np.fromstring(specific_row, dtype=float, sep=',')
    #             specific_rows.append(specific_row)
    #     return np.array(specific_rows)

    # def calculate_byte_offset(self,line_number, row_size):
    #     # Calculate the byte offset based on line number and row size
    #     def get_digit_count(number):
    #     # Calculate the number of digits in a number
    #         return len(str(number))
        
    #     count = get_digit_count(line_number)
    #     if count == 1:
    #         return line_number  * row_size + line_number
    #     offset = 0
    #     for i in range(count-1):
    #         if i == 0:
    #             offset = 10
    #         else:
    #             offset += (i+1)* (10 ** (i+1) - 10 ** i)
    #     offset += count * (line_number-10**(count-1))
    #     # print("offset: ",offset)
    #     return line_number * row_size+offset
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
        print("==================== In IVF Train =================")
        if not self.load:
            xp=self.fetch_from_binary(self.data_path,0,self.train_batch_size)
            print((xp)[:,0])
            embeds = xp[:, 1:]
            # embeds /= np.linalg.norm(embeds, axis=1, keepdims=True)
            
            embeds = np.array([record/np.linalg.norm(record) for record in embeds])
            (centroids, assignments) = kmeans2(embeds, self.centroids_num, self.iter,minit='points')
            self.centroids=centroids
            save_file(self.centroid_path, self.centroids)
            clustering_batch=self.preprocessing(xp,assignments)
            return clustering_batch

        else:
            self.centroids = load(self.centroid_path)

    #clustering_data
    def IVF_predict(self):
        print("==================== In IVF Predict =================")
        xp=self.fetch_from_binary(self.data_path,self.train_batch_size+self.predict_batch_size*self.prediction_count,self.predict_batch_size)
        embeds = xp[:, 1:]
        start = time.time()
        # embeds /= np.linalg.norm(embeds, axis=1, keepdims=True)
        embeds = np.array([record/np.linalg.norm(record) for record in embeds])
        print("normalization time: ", time.time()-start)
        self.prediction_count+=1
        start = time.time()
        assignments, _ = vq(embeds, self.centroids)
        print("vq time: ", time.time()-start)
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
        clusters = self.load_from_binary_file("ivf_cluster_"+str(0)+".bin")
        for c in nearset_centers:
                clusters = self.load_from_binary_file("ivf_cluster_"+str(c)+".bin")
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
                    with open("ivf_cluster_"+str(i)+".bin", 'ab') as file:
                    # print(item)
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
        
# from typing_extensions import runtime
@dataclass
class Result:
    run_time: float
    top_k: int
    db_ids: List[int]
    actual_ids: List[int]
def run_queries(top_k, num_runs):
    results = []
    ivfindex=ivf(data_path="saved_db.csv",train_batch_size=1000000,predict_batch_size= 10000,iter=32,centroids_num= 1024,nprops=64)
    train_batch_clusters=ivfindex.IVF_train()

    for _ in range(num_runs):
        query = np.random.random((1,70))
        query = query/np.linalg.norm(query)
        tic = time.time()
        # Clustering
        db_ids=ivfindex.IVF_test(query,train_batch_clusters)
        toc = time.time()
        run_time = toc - tic
        print("time of search:")
        print(run_time)
        # print(db_ids)
        np_rows=ivfindex.fetch_from_csv("saved_db.csv",1,1000000)
        embeds = np_rows[:, 1:]
        np_rows = np.array([record/np.linalg.norm(record) for record in embeds])
        tic = time.time()
        actual_ids = np.argsort(np_rows.dot(query.T).T / (np.linalg.norm(np_rows, axis=1) * np.linalg.norm(query)), axis= 1).squeeze().tolist()[::-1]
        toc = time.time()
        np_run_time = toc - tic
        # print(actual_ids[:30])
        results.append(Result(run_time, top_k, db_ids, actual_ids))
    return results

def eval(results: List[Result]):
    # scores are negative. So getting 0 is the best score.
    scores = []
    run_time = []
    counter = 0
    for res in results:
        run_time.append(res.run_time)
        # case for retireving number not equal to top_k, socre will be the lowest
        if len(set(res.db_ids)) != res.top_k or len(res.db_ids) != res.top_k:
            scores.append( -1 * len(res.actual_ids) * res.top_k)
            continue
        score = 0
        for id in res.db_ids:
            try:
                ind = res.actual_ids.index(id)
                if ind > res.top_k * 3:
                    score -= ind
                else :
                    counter += 1
            except:
                score -= len(res.actual_ids)
        scores.append(score)
    print(counter)
    return sum(scores) / len(scores), sum(run_time)/len(run_time)



