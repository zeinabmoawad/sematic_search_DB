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

class VecDBIVF:
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
            # self.centroid_path = "centroids.pkl"
            # if(self.data_size=="100k"):
            #     self.data_size=100000
            #     # change name of index file to load 100k indexer
            #     self.ivf_path = "ivf_100k"
            # elif(self.data_size=="1m"):
            #     self.data_size=1000000
            #     # change name of index file to load 1m indexer
            #     self.ivf_path = "ivf_1m"
            # elif(self.data_size=="5m"):
            #     self.data_size=5000000
            #     # change name of index file to load 5m indexer
            #     self.ivf_path = "ivf_5m"
            # elif(self.data_size=="10m"):
            #     self.data_size=10000000
            #     # change name of index file to load 10m indexer
            #     self.ivf_path = "ivf_10m"
            # elif(self.data_size=="15m"):
            #     self.data_size=15000000
            #     # change name of index file to load 15m indexer
            #     self.ivf_path = "ivf_15m"
            # else:
            #     self.data_size=20000000
            #     # change name of index file to load 20m indexer
            #     self.ivf_path = "ivf_20m"
            # self.ivfindex=ivf(data_path=self.file_path,folder_path = self.ivf_path,train_batch_size=0,predict_batch_size=0,iter=0,centroids_num=0,nprops=0,load=True)
            
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
        
        self.ivfindex.IVF_search_small_data(query=query,top_k=top_k)    
        
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
                self.ivfindex=ivf(data_path=self.file_path,folder_path="ivf_10k/",train_batch_size=train_batch_size,predict_batch_size=predict_batch_size,iter=iter,centroids_num= centroids_num,nprops=nprops)
                #Training
                cluster=self.ivfindex.IVF_train()
                self.ivfindex.add_clusters(cluster)
            elif(self.data_size==100000):
                os.makedirs("ivf_100k", exist_ok=True)
                train_batch_size=100000
                predict_batch_size=0
                centroids_num=128
                nprops=32
                iter=32
                self.ivfindex=ivf(data_path=self.file_path,folder_path="ivf_100k/",train_batch_size=train_batch_size,predict_batch_size=predict_batch_size,iter=iter,centroids_num= centroids_num,nprops=nprops)
                # Training
                cluster=self.ivfindex.IVF_train()
                self.ivfindex.add_clusters(cluster)
        else:
            if(self.data_size==1000000):
                os.makedirs("ivf_1m", exist_ok=True)
                train_batch_size=100000
                predict_batch_size=100000
                centroids_num=256
                nprops=32
                iter=32
                self.ivfindex=ivf(data_path=self.file_path,folder_path="ivf_1m/",train_batch_size=train_batch_size,predict_batch_size=predict_batch_size,iter=iter,centroids_num= centroids_num,nprops=nprops)
                # Training
                cluster=self.ivfindex.IVF_train()
                self.ivfindex.add_clusters(cluster)
                for i in range(9):
                    cluster=self.ivfindex.IVF_predict()
                    self.ivfindex.add_clusters(cluster)

            elif(self.data_size==5000000):
                os.makedirs("ivf_5m", exist_ok=True)
                train_batch_size=500000
                predict_batch_size=500000
                centroids_num=512
                nprops=64
                iter=32
                self.ivfindex=ivf(data_path=self.file_path,folder_path="ivf_5m/",train_batch_size=train_batch_size,predict_batch_size=predict_batch_size,iter=iter,centroids_num= centroids_num,nprops=nprops)
                # Training
                cluster=self.ivfindex.IVF_train()
                self.ivfindex.add_clusters(cluster)
                for i in range(9):
                    cluster=self.ivfindex.IVF_predict()
                    self.ivfindex.add_clusters(cluster)
            elif(self.data_size==15000000):
                os.makedirs("ivf_15m", exist_ok=True)
                train_batch_size=1500000
                predict_batch_size=1500000
                centroids_num=1024
                nprops=128
                iter=32
                self.ivfindex=ivf(data_path=self.file_path,folder_path="ivf_15m/",train_batch_size=train_batch_size,predict_batch_size=predict_batch_size,iter=iter,centroids_num= centroids_num,nprops=nprops)
                # Training
                cluster=self.ivfindex.IVF_train()
                self.ivfindex.add_clusters(cluster)
                for i in range(9):
                    cluster=self.ivfindex.IVF_predict()
                    self.ivfindex.add_clusters(cluster)
            elif(self.data_size==20000000):
                os.makedirs("ivf_20m", exist_ok=True)
                train_batch_size=1000000
                predict_batch_size=1000000
                centroids_num=1024
                nprops=128
                iter=64
                self.ivfindex=ivf(data_path=self.file_path,folder_path="ivf_20m/",train_batch_size=train_batch_size,predict_batch_size=predict_batch_size,iter=iter,centroids_num= centroids_num,nprops=nprops)
                # Training
                cluster=self.ivfindex.IVF_train()
                self.ivfindex.add_clusters(cluster)
                for i in range(19):
                    cluster=self.ivfindex.IVF_predict()
                    self.ivfindex.add_clusters(cluster)

        # end time
        end = time.time()
        print("time to build index = ", end - start)

        pass


