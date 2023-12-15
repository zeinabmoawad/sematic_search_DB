from typing import Dict, List
import numpy as np
from PQ import CustomIndexPQ
from ivf import ivf
import time
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
                self.ivf_path = "ivf_100k/"
                self.pq_path = "pq_100k/estimator.pkl"
                self.codes_path = "pq_100k/"
                self.ivfindex=ivf(data_path=file_path,folder_path= self.ivf_path,train_batch_size=100000,predict_batch_size=0,iter=32,centroids_num=128,nprops=32,load=True)
                self.pqindex = CustomIndexPQ( d = 70,m = 10,nbits = 7,path_to_db= file_path,load=True,
                                    estimator_file=self.pq_path,codes_file=self.codes_path,train_batch_size=100000,predict_batch_size=1000)
            elif(self.data_size=="1m"):
                self.data_size=1000000
                # change name of index file to load 1m indexer
                self.ivf_path = "ivf_1m/"
                self.pq_path = "pq_1m/estimator.pkl"
                self.codes_path = "pq_1m/"
                self.ivfindex=ivf(data_path=file_path,folder_path= self.ivf_path,train_batch_size=100000,predict_batch_size=100000,iter=32,centroids_num=1024,nprops=64,load=True)
                self.pqindex = CustomIndexPQ( d = 70,m = 10,nbits = 7,path_to_db= file_path,load=True,
                                    estimator_file=self.pq_path,codes_file=self.codes_path,train_batch_size=100000,predict_batch_size=1000)
            
            elif(self.data_size=="5m"):
                self.data_size=5000000
                # change name of index file to load 5m indexer
                self.ivf_path = "ivf_5m/"
                self.pq_path = "pq_5m/estimator.pkl"
                self.codes_path = "pq_5m/"
                self.ivfindex=ivf(data_path=file_path,folder_path= self.ivf_path,train_batch_size=100000,predict_batch_size=100000,iter=32,centroids_num=1024,nprops=64,load=True)
                self.pqindex = CustomIndexPQ( d = 70,m = 14,nbits = 8,path_to_db= file_path,load=True,
                                    estimator_file=self.pq_path,codes_file=self.codes_path,train_batch_size=100000,predict_batch_size=1000)
            
            elif(self.data_size=="10m"):
                self.data_size=10000000
                # change name of index file to load 10m indexer
                self.ivf_path = "ivf_10m/"
                self.pq_path = "pq_10m/estimator.pkl"
                self.codes_path = "pq_10m/"
                self.ivfindex=ivf(data_path=file_path,folder_path= self.ivf_path,train_batch_size=100000,predict_batch_size=100000,iter=32,centroids_num=1024,nprops=64,load=True)
                self.pqindex = CustomIndexPQ( d = 70,m = 14,nbits = 8,path_to_db= file_path,load=True,
                                    estimator_file=self.pq_path,codes_file=self.codes_path,train_batch_size=100000,predict_batch_size=1000)
            
            elif(self.data_size=="15m"):
                self.data_size=15000000
                # change name of index file to load 15m indexer
                self.ivf_path = "ivf_15m/"
                self.pq_path = "pq_15m/estimator.pkl"
                self.codes_path = "pq_15m/"
                self.ivfindex=ivf(data_path=file_path,folder_path= self.ivf_path,train_batch_size=100000,predict_batch_size=100000,iter=128,centroids_num=2048,nprops=64,load=True)
                self.pqindex = CustomIndexPQ( d = 70,m = 14,nbits = 8,path_to_db= file_path,load=True,
                                    estimator_file=self.pq_path,codes_file=self.codes_path,train_batch_size=100000,predict_batch_size=1000)
            
            else:
                self.data_size=20000000
                # change name of index file to load 20m indexer
                self.ivf_path = "ivf_20m/"
                self.pq_path = "pq_20m/estimator.pkl"
                self.codes_path = "pq_20m/"
                self.ivfindex=ivf(data_path=file_path,folder_path= self.ivf_path,train_batch_size=100000,predict_batch_size=100000,iter=32,centroids_num=1024,nprops=64,load=True)
                self.pqindex = CustomIndexPQ( d = 70,m = 14,nbits = 8,path_to_db= file_path,load=True,
                                    estimator_file=self.pq_path,codes_file=self.codes_path,train_batch_size=100000,predict_batch_size=1000)
            
            
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
                # free memory
                del row

        # free memory
        del rows
        self._build_index()

    # def retrive(self, query: Annotated[List[float], 70], top_k = 5):
    def retrive(self, query,top_k = 5):
        
        if(self.data_size<100000):
            return self.ivfindex.IVF_search_small_data(query=query,top_k=top_k)    
        elif(self.data_size<30000000):
          centroids = self.ivfindex.IVF_search_combo_data(query=query)
          return self.pqindex.search_using_IVF(query,centroids,top_k)
        
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
            self.ivfindex=ivf(data_path=self.file_path,folder_path = "ivf_10k/",train_batch_size=train_batch_size,predict_batch_size=predict_batch_size,iter=iter,centroids_num= centroids_num,nprops=nprops)
            #Training
            cluster=self.ivfindex.IVF_train()
            self.ivfindex.add_clusters(cluster)
          #100000
          elif(self.data_size==100000):
            os.makedirs("ivf_100k", exist_ok=True)
            train_batch_size=100000
            predict_batch_size=0
            centroids_num=128
            nprops=32
            iter=32
            self.ivfindex=ivf(data_path=self.file_path,folder_path = "ivf_100k/",train_batch_size=train_batch_size,predict_batch_size=predict_batch_size,iter=iter,centroids_num= centroids_num,nprops=nprops)
            # Training
            cluster=self.ivfindex.IVF_train()
            self.ivfindex.add_clusters(cluster)

        else:
          if(self.data_size==1000000):
            os.makedirs("ivf_1m", exist_ok=True)
            os.makedirs("pq_1m", exist_ok=True)
            self.ivfindex=ivf(data_path=self.file_path,folder_path = "ivf_1m/",train_batch_size=100000,predict_batch_size=100000,iter=64,centroids_num=1024,nprops=64)
            self.pqindex = CustomIndexPQ( d = 70,m = 10,nbits = 7,path_to_db= self.file_path,
                                    estimator_file="pq_1m/estimator.pkl",codes_file="pq_1m/",train_batch_size=100000,predict_batch_size=1000)
            
            # Training
            cluster=self.ivfindex.IVF_train()
            self.pqindex.train()
            self.pqindex.add(cluster)
            for i in range(9):
                cluster=self.ivfindex.IVF_predict()
                self.pqindex.add(cluster)

          elif(self.data_size==2000000):
            os.makedirs("ivf_2m", exist_ok=True)
            os.makedirs("pq_2m", exist_ok=True)
            self.ivfindex=ivf(data_path=self.file_path,folder_path = "ivf_2m/",train_batch_size=200000,predict_batch_size=200000,iter=64,centroids_num=512,nprops=64)
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
            os.makedirs("ivf_5m", exist_ok=True)
            os.makedirs("pq_5m", exist_ok=True)
            
            self.ivfindex=ivf(data_path=self.file_path,folder_path = "ivf_5m/",train_batch_size=500000,predict_batch_size=500000,iter=100,centroids_num=1024,nprops=64)
            self.pqindex = CustomIndexPQ( d = 70,m = 14,nbits = 8,path_to_db= self.file_path,
                                    estimator_file="pq_5m/estimator.pkl",codes_file="pq_5m/",train_batch_size=500000,predict_batch_size=1000)
            # Training
            cluster=self.ivfindex.IVF_train()
            self.pqindex.train()
            self.pqindex.add(cluster)
            for i in range(9):
                cluster=self.ivfindex.IVF_predict()
                self.pqindex.add(cluster)
          elif(self.data_size==10000000):
            # create folder for ivf and pq
            os.makedirs("ivf_10m", exist_ok=True)
            os.makedirs("pq_10m", exist_ok=True)
            
            self.ivfindex=ivf(data_path=self.file_path,folder_path = "ivf_10m/",train_batch_size=500000,predict_batch_size=500000,iter=100,centroids_num=1024,nprops=64)
            self.pqindex = CustomIndexPQ( d = 70,m = 14,nbits = 8,path_to_db= self.file_path,
                                    estimator_file="pq_10m/estimator.pkl",codes_file="pq_10m/",train_batch_size=500000,predict_batch_size=1000)
            # Training
            cluster=self.ivfindex.IVF_train()
            self.pqindex.train()
            self.pqindex.add(cluster)
            # calculate number of iterations if data size is 15m and train batch size is 1m and predict batch size is 500k
            for i in range(19):
                cluster=self.ivfindex.IVF_predict()
                self.pqindex.add(cluster)
          elif(self.data_size==15000000):
            # create folder for ivf and pq
            os.makedirs("ivf_15m", exist_ok=True)
            os.makedirs("pq_15m", exist_ok=True)
            
            self.ivfindex=ivf(data_path=self.file_path,folder_path = "ivf_15m/",train_batch_size=500000,predict_batch_size=500000,iter=100,centroids_num=2048,nprops=64)
            self.pqindex = CustomIndexPQ( d = 70,m = 14,nbits = 8,path_to_db= self.file_path,
                                    estimator_file="pq_15m/estimator.pkl",codes_file="pq_15m/",train_batch_size=500000,predict_batch_size=1000)
            # Training
            cluster=self.ivfindex.IVF_train()
            self.pqindex.train()
            self.pqindex.add(cluster)
            # calculate number of iterations if data size is 15m and train batch size is 1m and predict batch size is 500k
            for i in range(29):
                cluster=self.ivfindex.IVF_predict()
                self.pqindex.add(cluster)
          elif(self.data_size==20000000):
            # create folder for ivf and pq
            os.makedirs("ivf_20m", exist_ok=True)
            os.makedirs("pq_20m", exist_ok=True)
            
            self.ivfindex=ivf(data_path=self.file_path,folder_path = "ivf_20m/",train_batch_size=500000,predict_batch_size=500000,iter=100,centroids_num=2048,nprops=64)
            self.pqindex = CustomIndexPQ( d = 70,m = 14,nbits = 8,path_to_db= self.file_path,
                                    estimator_file="pq_20m/estimator.pkl",codes_file="pq_20m/",train_batch_size=500000,predict_batch_size=1000)
            # Training
            cluster=self.ivfindex.IVF_train()
            self.pqindex.train()
            self.pqindex.add(cluster)
            # calculate number of iterations if data size is 15m and train batch size is 1m and predict batch size is 500k
            for i in range(39):
                cluster=self.ivfindex.IVF_predict()
                self.pqindex.add(cluster)
    
        # end time
        end = time.time()
        print("time to build index = ", end - start)

        pass


