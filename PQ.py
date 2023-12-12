from __future__ import annotations

import logging

import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics.pairwise import euclidean_distances
from sklearn.metrics.pairwise import cosine_similarity
import pickle
import os
import sklearn.metrics.pairwise as pw
import math
from sklearn import preprocessing
from scipy.cluster.vq import kmeans2,vq
import heapq
# !python evaluation.py
import platform
import struct



logger = logging.getLogger(__name__)

BITS2DTYPE = {
    8: np.uint8,
    4: np.uint8,
    1: np.uint8,
    2: np.uint8,
    3: np.uint8,
    5: np.uint8,
    6: np.uint8,
    7: np.uint8,
    9: np.uint16,
    10: np.uint16,
    11: np.uint16
}

def save_file(file_path,file_save):
    with open(file_path, 'wb') as file:
        pickle.dump(file_save, file)


def load(file_path):
    with open(file_path, 'r') as file:
        file_loaded = pickle.load(file)
    return file_loaded

class CustomIndexPQ:
    """Custom IndexPQ implementation.

    Parameters
    ----------
    d
        Dimensionality of the original vectors.

    m
        Number of segments.

    nbits
        Number of bits.

    estimator_kwargs
        Additional hyperparameters passed onto the sklearn KMeans
        class.

    """
    

    def __init__(
        self,
        d: int,
        m: int,
        nbits: int,
        estimator_file:str,
        codes_file:str,
        path_to_db: str,
        train_batch_size:int,
        predict_batch_size: int,
        **estimator_kwargs: str | int
    ) -> None:
        if d % m != 0:
            raise ValueError("d needs to be a multiple of m")

        if nbits not in BITS2DTYPE:
            raise ValueError(f"Unsupported number of bits {nbits}")

        self.m = m
        self.k = 2**nbits
        self.d = d
        self.ds = d // m
        self.estimator_file = estimator_file
        self.codes_file = codes_file
        self.path_to_db = path_to_db
        self.train_batch_size = train_batch_size
        self.predict_batch_size=predict_batch_size
        self.prediction_count=0

        # self.estimators = [
        #     KMeans(n_clusters=self.k, **estimator_kwargs) for _ in range(m)
        # ]
        # logger.info(f"Creating following estimators: {self.estimators[0]!r}")
        # save_file(self.estimator_file,self.estimators) #"estimators.pkl"

        self.is_trained = False
        self.ids = None

        self.dtype = BITS2DTYPE[nbits]
        self.dtype_orig = np.float32

        self.codes: np.ndarray | None = None

    def load_db(self):
        # load csv file as numpy array
        embeds = np.loadtxt(self.path_to_db, delimiter=",")
        # get the id column
        ids = embeds[:, 0]
        # get the embed columns
        embeds = embeds[:, 1:]
        return embeds,ids


    def fetch_from_binary(self,file_path,line,size):
      with open(file_path, "rb") as binary_file:
        # Read an integer (4 bytes) from the current position
        binary_file.seek(line*564)
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

    def calculate_byte_offset(self,line_number, row_size):
        # Calculate the byte offset based on line number and row size
        def get_digit_count(number):
        # Calculate the number of digits in a number
            return len(str(number))
        line_number = int(line_number)
        count = get_digit_count(line_number)
        if count == 1:
            return line_number  * row_size + line_number
        offset = 0
        for i in range(count-1):
            if i == 0:
                offset = 10
            else:
                offset += (i+1)* (10 ** (i+1) - 10 ** i)
        offset += count * (line_number-10**(count-1))
        return line_number * row_size+offset
    def train(self) -> None:
        """Train all KMeans estimators.

        Parameters
        ----------
        X
            Array of shape `(n, d)` and dtype `float32`.

        """
        if self.is_trained:
            raise ValueError("Training multiple times is not allowed")

        # load data from csv file
        print("start training")
        data = np.array(self.fetch_from_binary(self.path_to_db,0,self.train_batch_size))
        # print(data)
        X= data[:, 1:]

        self.estimators = [] 
        for i in range(self.m):
            # estimator = self.estimators[i]
            X_i = X[:, i * self.ds : (i + 1) * self.ds]
            centroids, _ = kmeans2(X_i, self.k,iter=100,minit='points') 
            print("Finished KMeans for the ",i,"-th segment"," from ",self.m," segments")
            logger.info(f"Fitting KMeans for the {i}-th segment")
            # estimator.fit(X_i)
            # print("centroids = ",centroids)
            self.estimators.append(centroids)
        save_file(self.estimator_file,self.estimators)
        # save_file('estimators.txt',self.estimators)

        self.is_trained = True
        # save estimators to csv file
        print("finished training")


    def encode(self) -> np.ndarray:
        """Encode original features into codes.

        Parameters
        ----------
        X
            Array of shape `(n_queries, d)` of dtype `np.float32`.

        Returns
        -------
        result
            Array of shape `(n_queries, m)` of dtype `np.uint8`.
        """
        # create 2d array
        result = []

        # loop over each row in csv file
        print("start encoding")
        xp=self.fetch_from_binary(self.path_to_db,self.train_batch_size+self.predict_batch_size*self.prediction_count,self.predict_batch_size)
        id = int(float(xp[:,0]))
        # print("id = ", np.array([id]))
        X = [float(e) for e in xp[:, 1:]]
        X = np.array(X).reshape(1,self.d)
        code = np.empty((1, self.m), dtype=self.dtype)
        for i in range(self.m):
            estimator = self.estimators[i]
            X_i = X[:, i * self.ds : (i + 1) * self.ds]
            code_i, _ = vq(X_i, estimator) 
            # code[:, i] = estimator.predict(X_i)
            code[:, i] = code_i
            # add id to code at beginning
        # code = np.concatenate((np.array([id]).reshape(1,1),code), axis=1).astype(np.int32)
        # append to result
        # print("code  ",code)
        result.append(code)
        # convert result to numpy array of shape (n,m) instead of list of shape (n,1,m)
        # print("resuls = ",result)
        result = np.array(result).reshape(-1,self.m )
        print("finished encoding")
        return result
    def encode_using_IVF(self,X: np.ndarray) -> np.ndarray:
        # create 2d array
        result = []

        # loop over each row in csv file
        for row in X:
            id = int(row[0])
            X = row[1]
            X = np.array(X).reshape(1,self.d)
            code = np.empty((1, self.m), dtype=self.dtype)
            for i in range(self.m):
                estimator = self.estimators[i]
                X_i = X[:, i * self.ds : (i + 1) * self.ds]
                code_i, _ = vq(X_i, estimator) 
                # code[:, i] = estimator.predict(X_i)
                code[:, i] = code_i
                # add id to code at beginning 
            code = np.concatenate((np.array([id]).reshape(1,1),code), axis=1).astype(np.int32)
            # print("code shape = ",code.shape)
            # code = np.concatenate((np.array([id]).reshape(1,1),code), axis=1).astype(np.int32)
            # append to result
            # print("code  ",code)
            result.append(code)
        # convert result to numpy array of shape (n,m) instead of list of shape (n,1,m)
        # print("resuls = ",result)
        result = np.array(result).reshape(-1,self.m+1)
        return result
    def add(self,data:np.ndarray= None) -> None:
        """Add vectors to the database (their encoded versions).

        Parameters
        ----------
        X
            Array of shape `(n_codes, d)` of dtype `np.float32`.
        """
        if not self.is_trained:
            raise ValueError("The quantizer needs to be trained first.")
        # self.codes = self.encode()
        if data is not None:
            for i in range(len(data)):
                # open pickle file with name i if exists or create new one and append to it
                result = self.encode_using_IVF(data[i])
                # print(result)
                # print(result)
                for item in result:
                    # print(item)
                    with open("codes_"+str(i)+".txt", 'a') as file:
                        line = f"{item[0]} {' '.join(map(str, item[1:]))}\n"
                        file.write(line)

        else:
            save_file(self.codes_file,self.encode()) #"codes.pkl"

    def compute_asymmetric_distances(self, X: np.ndarray) -> np.ndarray:
        """Compute asymmetric distances to all database codes.

        Parameters
        ----------
        X
            Array of shape `(n_queries, d)` of dtype `np.float32`.

        Returns
        -------
        distances
            Array of shape `(n_queries, n_codes)` of dtype `np.float32`.

        """
        if not self.is_trained:
            raise ValueError("The quantizer needs to be trained first.")
        # codes = load(self.codes_file)
        if self.codes is None:
            raise ValueError("No codes detected. You need to run `add` first")

        n_queries = len(X)
        n_codes = len(self.codes)

        distance_table = np.empty(
            (n_queries, self.m, self.k), dtype=self.dtype_orig
        )  # (n_queries, m, k)

        
        for i in range(self.m):
            X_i = X[:, i * self.ds : (i + 1) * self.ds]  # (n_queries, ds)
            centers = self.estimators[i] # (k, ds)
            distance_table[:, i, :] = euclidean_distances(
                X_i, centers, squared=True
            )
        #     X_i = X_i / np.linalg.norm(X_i, axis=1,keepdims=True)
        #     # cosine_similarity = np.dot(X_i,centers)/np.linalg.norm(X) * np.linalg.norm(centers)
        # #     distance_table[:, i, :] = 2 - 2*cosine_similarity(X_i, centers)
        #     ans = self._cal_score(centers,X_i.reshape(-1,1))
        #     # b = []
        #     # for a in ans:
        #     #     b.append(math.sqrt(2*abs(1-a)))
        #     # print("b answer = ",b,)
        #     distance_table[:, i, :] = ans.reshape(1,-1)
            # print("of my function = ",ans.reshape(1,-1),"\n")
        # # print("of function defined = ",1 - cosine_similarity(X_i, centers),"\n")


        distances = np.zeros((n_queries, n_codes), dtype=self.dtype_orig)
        # print("codes shape = ",self.codes.shape)
        for i in range(self.m):
            distances += distance_table[:, i, self.codes[:, i+1]]
        # append ids column to distances
        distances = np.concatenate((np.array(self.codes[:,0]).reshape(-1,1),distances.reshape(-1,1)), axis=1).astype(self.dtype_orig)

        return distances

    def search(self, X: np.ndarray, k: int) -> tuple[np.ndarray, np.ndarray]:
        """Find k closest database codes to given queries.

        Parameters
        ----------
        X
            Array of shape `(n_queries, d)` of dtype `np.float32`.

        k
            The number of closest codes to look for.

        Returns
        -------
        distances
            Array of shape `(n_queries, k)`.

        indices
            Array of shape `(n_queries, k)`.
        """

        if self.codes is None:
            # load codes from pickle file
            self.codes = load(self.codes_file)
            # print("print(self.codes) =",self.codes)
            # print("Before ",self.codes.shape)
            # self.ids = self.codes[:,0]
            # self.codes = self.codes[:,1:]
            # print("After ",self.codes.shape)
        if self.estimators is None:
            # load estimators from pickle file
            self.estimators = load(self.estimator_file)

        # n_queries = len(X)
        # X = X / np.linalg.norm(X, axis=1,keepdims=True)
        distances_all = self.compute_asymmetric_distances(X)
        # print("Distance ",distances_all.shape)
        # print("Ids ",self.ids.shape)
        # print("Ids ",self.ids.reshape(-1,1).shape)
        # append ids column to distances_all
        # distances_all = np.concatenate((self.ids.reshape(-1,1),np.array(distances_all).reshape(-1,1)), axis=1).astype(self.dtype_orig)

        
        # sort distances_all by distances column not id column

        # distances_all = distances_all[distances_all[:,1].argsort()]
        # with open("test.txt", 'wb') as file:
            # np.savetxt(file,distances_all)
        
        # sort distances_all by distances column not id column
        indices = np.argsort(distances_all, axis=1)[:, :k]

        # distances = np.empty((n_queries, k), dtype=np.float32)
        # for i in range(n_queries):
            # distances[i] = distances_all[i][indices[i]]
        return indices[0]
        # return np.array(distances_all[:k,0]).astype(np.int32)

    def load_file(self,file):
        loaded_data = []
        # print(file.readlines())
        file_content = file.readlines()
        # print(file_content)
        for line in file_content:
            # print('line = ',line)
            values = line.strip().split(' ')
            # print(values)
            # code = np.concatenate((np.array([id]).reshape(1,1),code), axis=1).astype(np.int32)
            # array_values = np.array()
            float_data = [int(item) for item in values]
            # code = np.array(float_data)
                
            # array_values = np.array([int(x) for x in values])
            # print(array_values)
            # code = np.concatenate((np.array([id]).reshape(1,1),code), axis=1).astype(np.int32)
            loaded_data.append(float_data)
        return np.array(loaded_data)
    
    def search_using_IVF(self, Xq: np.ndarray,centriods:np.ndarray,k: int,refine:bool = True) -> tuple[np.ndarray, np.ndarray]:
        """Find k closest database codes to given queries.

        Parameters
        ----------
        X
            Array of shape `(n_queries, d)` of dtype `np.float32`.

        k
            The number of closest codes to look for.

        Returns
        -------
        distances
            Array of shape `(n_queries, k)`.

        indices
            Array of shape `(n_queries, k)`.
        """
        # self.codes = codes
        # load codes from pickle file with name i in centroid list
        if self.estimators is None:
            # load estimators from pickle file
            self.estimators = load(self.estimator_file)
        distances = []

        for i in range(len(centriods)):
            # open pickle file with name i if exists or create new one and append to it
            with open("codes_"+str(centriods[i])+".txt", 'r') as file:
                # if i == 0:
                #     self.codes = pickle.load(file)
                # else:
                #     self.codes = np.concatenate((self.codes,pickle.load(file)), axis=0)
                self.codes = self.load_file(file)
                # print("hereee")
                distances_all = self.compute_asymmetric_distances(Xq)
                # print("Distance ",distances_all.shape)
                distances_all = distances_all[distances_all[:,1].argsort()][:k,:]
                # indices = np.argsort(distances_all, axis=1)[:, :k]
                if i == 0:
                    distances = distances_all
                else:
                    distances = np.concatenate((distances,distances_all), axis=0)
        

        if refine:
            # call refinement
            return self.refinement(distances[:,0],Xq,k)
        
        # sort distances_all by distances column not id column
        distances = distances[distances[:,1].argsort()][:k,:]

        # return id column of distances
        return distances[:,0].astype(np.int32)
    def refinement(self,IDs,query,k):
        """
        refinement to output vectors
        1- for 1000 IDs select their vectors from csv or db
        2- compute cosine similarity with query vector
        3- sort output and retieve best k IDs
        """
        # get vectors of corresponding ids
        refine_vectors = []
        for i in range(len(IDs)):
            # get vector of corresponding id
            vec = self.fetch_from_binary(self.path_to_db,int(IDs[i]),1)

            # convert to 1d array
            vec = np.array(vec).reshape(-1)

            # remove id from vec
            vec = vec[1:]

            # convert to 1d array
            # vec = np.array(vec).reshape(1,-1)
            # compute cosine similarity with query
            cosine_similarity_output = self._cal_score(vec, query.reshape(-1))

            # append cosine_similarity_output 
            refine_vectors.append([IDs[i],cosine_similarity_output])
        
        # covert to numpy array
        refine_vectors = np.array(refine_vectors)

        ids = self.top_k_largest_elements(refine_vectors,k)
        return ids
        # sort on cosine similarity
        refine_vectors = refine_vectors[refine_vectors[:,1].argsort()]

        # reverse to get highest cosine similarity
        refine_vectors = refine_vectors[::-1]

        # get top k IDs
        refine_vectors = refine_vectors[:k]

        # get IDs
        ids2 = refine_vectors[:,0].astype(np.int32)
        
        # compare ids if not equal assert
        assert np.array_equal(ids,ids2)

        return refine_vectors[:,0].astype(np.int32)
    
    def _cal_score(self, vec1, vec2):
        dot_product = np.dot(vec1, vec2)
        norm_vec1 = np.linalg.norm(vec1)
        norm_vec2 = np.linalg.norm(vec2)
        cosine_similarity = dot_product / (norm_vec1 * norm_vec2)
        return cosine_similarity



    def top_k_largest_elements(self, nums, k):

        indices = heapq.nlargest(k, range(len(nums)), key=lambda i: nums[i, 1])
        # return ids of top k largest elements from list of indices
        return nums[indices,0].astype(np.int32)
        min_heap = []
        
        # Push the first k elements onto the heap
        for num in nums[:k]:
            heapq.heappush(min_heap, (-1*num[1],num[0]))
        
        # Continue pushing elements onto the heap while maintaining size k
        for num in nums[k:]:
            if num[1]*-1 < min_heap[0][0]:
                heapq.heappop(min_heap)
                heapq.heappush(min_heap, (num[1],num[0]))
        
        # The heap now contains the top k largest elements
        # get the top k largest elements in sorted order
        return [heapq.heappop(min_heap)[1] for _ in range(len(min_heap))]
        # return list(min_heap)



        # def add_clusters(self,cluster:np.ndarray= None) -> None:
        # """Add vectors to the database (their encoded versions).

        # Parameters
        # ----------
        # X
        #     Array of shape (n_codes, d) of dtype np.float32.
        # """
        # if cluster is not None:
        #     for i in range(len(cluster)):
        #         for item in cluster[i]:
        #             with open("ivf_cluster_"+str(i)+".txt", 'a') as file:
        #                 line = f"{item[0]} {' '.join(map(str, item[1]))}\n"
        #                 file.write(line)
            
    # def load_from_text_file(self,file_path):
    #     """
    #     Load data from a text file.

    #     Parameters:
    #     - file_path: Path to the file.

    #     Returns:
    #     - List of tuples loaded from the file.
    #     """
    #     loaded_data = []
    #     with open(file_path, 'r') as file:
    #         for line in file.readlines():
    #             values = line.strip().split(' ')
    #             array_values = np.array([float(x) for x in values])
    #             loaded_data.append((array_values[0], array_values[1:]))
    #     return loaded_data