from typing import Dict, List
# from typing import Annotated
import numpy as np
from PQ import CustomIndexPQ
from ivf import ivf
from HNSW import HNSW
import time
# import faiss
import struct
import os

class VecDB:
    def __init__(self, file_path = "saved_db.bin", new_db = True) -> None:
        self.file_path = file_path
        self.data_size = 0
        if new_db:
            # just open new file to delete the old one
            with open(self.file_path, "w") as fout:
                # if you need to add any head to the file
                pass
        else:
            # load index from file
            # create constructor for ivf and pq
            # extract size of data from file name "saved_db_100k.bin" or "saved_db_1m.bin" or "saved_db_5m.bin" 
            # find second last index of "_" and last index of "."
            self.data_size = self.file_path[self.file_path.rfind("_")+1:self.file_path.rfind(".")]
            if(self.data_size=="100k"):
                self.data_size=100000
                # change name of index file to load 100k indexer
                self.ivf_path = "ivf_100k/centroids.pkl"
                self.pq_path = "pq_100k/estimator.pkl"
                self.codes_path = "pq_100k/"
            elif(self.data_size=="1m"):
                self.data_size=1000000
                # change name of index file to load 1m indexer
                self.ivf_path = "ivf_1m/centroids.pkl"
                self.pq_path = "pq_1m/estimator.pkl"
                self.codes_path = "pq_1m/"
            elif(self.data_size=="5m"):
                self.data_size=5000000
                # change name of index file to load 5m indexer
                self.ivf_path = "ivf_5m/centroids.pkl"
                self.pq_path = "pq_5m/estimator.pkl"
                self.codes_path = "pq_5m/"
            elif(self.data_size=="10m"):
                self.data_size=10000000
                # change name of index file to load 10m indexer
                self.ivf_path = "ivf_10m/centroids.pkl"
                self.pq_path = "pq_10m/estimator.pkl"
                self.codes_path = "pq_10m/"
            elif(self.data_size=="15m"):
                self.data_size=15000000
                # change name of index file to load 15m indexer
                self.ivf_path = "ivf_15m/centroids.pkl"
                self.pq_path = "pq_15m/estimator.pkl"
                self.codes_path = "pq_15m/"
            else:
                self.data_size=20000000
                # change name of index file to load 20m indexer
                self.ivf_path = "ivf_20m/centroids.pkl"
                self.pq_path = "pq_20m/estimator.pkl"
                self.codes_path = "pq_20m/"
            
            self.ivfindex=ivf(data_path=self.file_path,centroid_path = self.ivf_path,train_batch_size=100000,predict_batch_size=100000,iter=32,centroids_num=256,nprops=32)
            self.pqindex = CustomIndexPQ( d = 70,m = 10,nbits = 7,path_to_db= self.file_path,load=True,
                                    estimator_file=self.pq_path,codes_file=self.codes_file,train_batch_size=100000,predict_batch_size=1000)
            
            
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
                self.writing_binary_file( self.file_path,row["id"],row["embed"])

        print("inserted successfully")
        self._build_index()

    # def retrive(self, query: Annotated[List[float], 70], top_k = 5):
    def retrive(self, query,top_k = 5):
        print("================In Search=====================")
        
        if(self.data_size<1000000):
            return self.ivfindex.IVF_search_small_data(query=query,top_k=top_k)    
        elif(self.data_size<5000000):
          centroids = self.ivfindex.IVF_search_combo_data(query=query)
          return self.pqindex.search_using_IVF(query,centroids,top_k)
        else:
          centroids = self.HNSW.HNSW_search(query)
          return self.pqindex.search_using_IVF(query,centroids[0],top_k)
        
    def _build_index(self):
        # start time
        start = time.time()
        if(self.data_size<1000000):
          #10000
          if(self.data_size==10000):
            os.makedirs("ivf_10k", exist_ok=True)
            train_batch_size=10000
            predict_batch_size=0
            centroids_num=16
            nprops=4
            iter=32
            self.ivfindex=ivf(data_path=self.file_path,centroid_path = "ivf_10k/centroids.pkl",train_batch_size=train_batch_size,predict_batch_size=predict_batch_size,iter=iter,centroids_num= centroids_num,nprops=nprops)
            #Training
            cluster=self.ivfindex.IVF_train()
            self.ivfindex.add_clusters(cluster)
          #100000
          elif(self.data_size==100000):
            os.makedirs("ivf_100k", exist_ok=True)
            os.makedirs("pq_100k", exist_ok=True)
            train_batch_size=100000
            predict_batch_size=0
            centroids_num=128
            nprops=32
            iter=32
            self.ivfindex=ivf(data_path=self.file_path,centroid_path = "ivf_100k/centroids.pkl",train_batch_size=train_batch_size,predict_batch_size=predict_batch_size,iter=iter,centroids_num= centroids_num,nprops=nprops)
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
            # print("data_sze=10000")
            os.makedirs("ivf_1m", exist_ok=True)
            os.makedirs("pq_1m", exist_ok=True)
            self.ivfindex=ivf(data_path=self.file_path,centroid_path = "ivf_1m/centroids.pkl",train_batch_size=100000,predict_batch_size=100000,iter=64,centroids_num=1024,nprops=64)
            self.pqindex = CustomIndexPQ( d = 70,m = 10,nbits = 7,path_to_db= self.file_path,
                                    estimator_file="pq_1m/estimator.pkl",codes_file="pq_1m/",train_batch_size=100000,predict_batch_size=1000)
            
            # self.HNSW = HNSW(self.ivfindex.nprops)

            # Training
            cluster=self.ivfindex.IVF_train()
            self.pqindex.train()
            self.pqindex.add(cluster)
            # self.HNSW.HNSW_train(self.ivfindex.centroids)
            for i in range(9):
                cluster=self.ivfindex.IVF_predict()
                self.pqindex.add(cluster)

          elif(self.data_size==2000000):
            os.makedirs("ivf_2m", exist_ok=True)
            os.makedirs("pq_2m", exist_ok=True)
            self.ivfindex=ivf(data_path=self.file_path,centroid_path = "ivf_2m/centroids.pkl",train_batch_size=200000,predict_batch_size=200000,iter=64,centroids_num=512,nprops=64)
            self.pqindex = CustomIndexPQ( d = 70,m = 14,nbits = 7,path_to_db= self.file_path,
                                    estimator_file="pq_2m/estimator.pkl",codes_file="pq_2m/",train_batch_size=200000,predict_batch_size=1000)
            # Training
            cluster=self.ivfindex.IVF_train()
            self.pqindex.train()
            self.pqindex.add(cluster)
            for i in range(9):
                cluster=self.ivfindex.IVF_predict()
                self.pqindex.add(cluster)

          elif(self.data_size==5000000):
            print("=========IN 5M=============")
            # create folder for ivf and pq
            os.makedirs("ivf_5m", exist_ok=True)
            os.makedirs("pq_5m", exist_ok=True)
            
            self.ivfindex=ivf(data_path=self.file_path,centroid_path = "ivf_5m/centroids.pkl",train_batch_size=500000,predict_batch_size=500000,iter=64,centroids_num=1024,nprops=128)
            self.pqindex = CustomIndexPQ( d = 70,m = 14,nbits = 7,path_to_db= self.file_path,
                                    estimator_file="pq_5m/estimator.pkl",codes_file="pq_5m/",train_batch_size=500000,predict_batch_size=1000)
            # Training
            cluster=self.ivfindex.IVF_train()
            self.pqindex.train()
            self.pqindex.add(cluster)
            for i in range(9):
                cluster=self.ivfindex.IVF_predict()
                self.pqindex.add(cluster)

    
        # end time
        end = time.time()
        print("time to build index = ", end - start)

        pass


