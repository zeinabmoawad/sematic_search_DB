from typing import Dict, List
# from typing import Annotated
import numpy as np
from PQ import CustomIndexPQ
from ivf import ivf
from HNSW import HNSW
import time
# import faiss
import struct

class VecDBWorst:
    def __init__(self, file_path = "saved_db.bin", new_db = True) -> None:
        self.file_path = file_path
        self.data_size = 0
        if new_db:
            # just open new file to delete the old one
            with open(self.file_path, "w") as fout:
                # if you need to add any head to the file
                pass
    
      # def insert_records(self, rows: List[Dict[int, Annotated[List[float], 70]]]):
    def writing_binary_file(self,file_path,id,row):
      # Open a binary file in write mode
      with open(file_path, "ab") as binary_file:
          # Write an integer (4 bytes)
          binary_file.write(struct.pack('i', id))  
          binary_file.write(struct.pack(f'{len(row)}d', *row))

    def insert_records(self, rows):
        self.data_size+=len(rows)
        with open(self.file_path, "a+") as fout:
            for row in rows:
                id, embed = row["id"], row["embed"]
                # row_str = f"{id}," + ",".join([str(e) for e in embed])
                # embeds = np.concatenate((np.array(id).reshape(1,1), np.array(embed).reshape(1,70)), axis=1).astype(np.float32)
                # save in csv
                # np.savetxt(fout, embeds, delimiter=",")
                # np.savetxt(fout, embeds, delimiter=",", fmt="%f")
                self.writing_binary_file( self.file_path,row["id"],row["embed"])
                # fout.write(f"{row_str}\n")
        print("inserted ",len(rows)," rows")
        # self._build_index()

    # def retrive(self, query: Annotated[List[float], 70], top_k = 5):
    def retrive(self, query,top_k = 5):
        print("================In Search=====================")
        # scores = []
        # with open(self.file_path, "r") as fin:
        #     for row in fin.readlines():
        #         row_splits = row.split(",")
        #         id = int(row_splits[0])
        #         embed = [float(e) for e in row_splits[1:]]
        #         score = self._cal_score(query, embed)
        #         scores.append((score, id))
        # # here we assume that if two rows have the same score, return the lowest ID
        # scores = sorted(scores)[:top_k]
        # return [s[1] for s in scores]
        
        if(self.data_size<1000000):
            return self.ivfindex.IVF_search_small_data(query=query,top_k=top_k)    
        elif(self.data_size<5000000):
          centroids = self.ivfindex.IVF_search_combo_data(query=query)
          return self.pqindex.search_using_IVF(query,centroids,top_k)
        else:
          centroids = self.HNSW.HNSW_search(query)
          return self.pqindex.search_using_IVF(query,centroids[0],top_k)
        # if(self.data_size<=1000000):
        #     _,indices = self.HNSW.search(query, self.ivfindex.nprops)
        #     print(indices)
        #     return np.array(indices)
            
    def _cal_score(self, vec1, vec2):
        dot_product = np.dot(vec1, vec2)
        norm_vec1 = np.linalg.norm(vec1)
        norm_vec2 = np.linalg.norm(vec2)
        cosine_similarity = dot_product / (norm_vec1 * norm_vec2)
        return cosine_similarity

    def _build_index(self):
        # start time
        start = time.time()
        if(self.data_size<1000000):
          #10000
          if(self.data_size==10000):
            train_batch_size=10000
            predict_batch_size=0
            centroids_num=16
            nprops=4
            iter=32
            self.ivfindex=ivf(data_path=self.file_path,train_batch_size=train_batch_size,predict_batch_size=predict_batch_size,iter=iter,centroids_num= centroids_num,nprops=nprops,folder_path="10000/",centroid_path="centroids.pkl")
            #Training
            cluster=self.ivfindex.IVF_train()
            self.ivfindex.add_clusters(cluster)
          #100000
          elif(self.data_size==100000):
            train_batch_size=100000
            predict_batch_size=0
            centroids_num=128
            nprops=32
            iter=32
            self.ivfindex=ivf(data_path=self.file_path,train_batch_size=train_batch_size,predict_batch_size=predict_batch_size,iter=iter,centroids_num= centroids_num,nprops=nprops)
            # Training
            cluster=self.ivfindex.IVF_train()
            self.ivfindex.add_clusters(cluster)
          #1000000
          # elif(self.data_size==1000000):
          #   train_batch_size=100000
          #   predict_batch_size=100000
          #   centroids_num=128
          #   nprops=32
          #   iter=32
          #   self.ivfindex=ivf(data_path=self.file_path,train_batch_size=train_batch_size,predict_batch_size=predict_batch_size,iter=iter,centroids_num= centroids_num,nprops=nprops)
          #   # Training
          #   cluster=self.ivfindex.IVF_train()
          #   self.ivfindex.add_clusters(cluster)
          #   for i in range(9):
          #       cluster=self.ivfindex.IVF_predict()
          #       self.ivfindex.add_clusters(cluster)
        else:
          #5000000 ,1000000 ,2000000
          if(self.data_size==1000000):
            print("data_sze=10000")
            self.ivfindex=ivf(data_path=self.file_path,train_batch_size=100000,predict_batch_size=100000,iter=32,centroids_num=256,nprops=32)
            self.pqindex = CustomIndexPQ( d = 70,m = 10,nbits = 7,path_to_db= self.file_path,
                                    estimator_file="estimator.pkl",codes_file="codes.pkl",train_batch_size=100000,predict_batch_size=1000)
            self.HNSW = HNSW(self.ivfindex.nprops)
            # Training
            cluster=self.ivfindex.IVF_train()
            self.pqindex.train()
            self.pqindex.add(cluster)
            self.HNSW.HNSW_train(self.ivfindex.centroids)
            for i in range(9):
                cluster=self.ivfindex.IVF_predict()
                self.pqindex.add(cluster)

          elif(self.data_size==2000000):
            self.ivfindex=ivf(data_path=self.file_path,train_batch_size=200000,predict_batch_size=200000,iter=64,centroids_num=512,nprops=64)
            self.pqindex = CustomIndexPQ( d = 70,m = 14,nbits = 7,path_to_db= self.file_path,
                                    estimator_file="estimator.pkl",codes_file="codes.pkl",train_batch_size=200000,predict_batch_size=1000)
            # Training
            cluster=self.ivfindex.IVF_train()
            self.pqindex.train()
            self.pqindex.add(cluster)
            for i in range(9):
                cluster=self.ivfindex.IVF_predict()
                self.pqindex.add(cluster)

          elif(self.data_size==5000000):
            print("=========IN 5M=============")
            self.ivfindex=ivf(data_path=self.file_path,train_batch_size=500000,predict_batch_size=500000,iter=64,centroids_num=1024,nprops=128)
            self.pqindex = CustomIndexPQ( d = 70,m = 14,nbits = 7,path_to_db= self.file_path,
                                    estimator_file="estimator.pkl",codes_file="codes.pkl",train_batch_size=500000,predict_batch_size=1000)
            self.HNSW = HNSW(self.ivfindex.nprops)
            # Training
            cluster=self.ivfindex.IVF_train()
            self.pqindex.train()
            self.pqindex.add(cluster)
            self.HNSW.HNSW_train(self.ivfindex.centroids)
            for i in range(9):
                cluster=self.ivfindex.IVF_predict()
                self.pqindex.add(cluster)

    
        # end time
        end = time.time()
        print("time to build index = ", end - start)

        pass


