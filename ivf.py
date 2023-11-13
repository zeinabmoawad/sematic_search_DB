import math
import numpy as np
from scipy.cluster.vq import kmeans2,vq
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
from sklearn import preprocessing
import time
# from  worst_case_implementation import VecDBWorst
from dataclasses import dataclass
from typing import List

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
        centroids_num
        ) -> None:
        self.train_batch_size=train_batch_size
        self.predict_batch_size=predict_batch_size
        self.data_path=data_path
        self.prediction_count=0
        # self.k=k
        self.nprops=nprops
        self.iter=iter
        self.centroids_num=centroids_num

    #Fetching file
    def fetch_from_csv(self,file_path,line_number,size):
        row_size = 80*8 #size of each row in bytes
        byte_offset = (line_number - 1) * row_size
        specific_rows=[]
        with open(file_path, 'r', encoding='utf-8') as csv_file:
            csv_file.seek(byte_offset)
            specific_row = csv_file.readline().strip()
            specific_row = np.fromstring(specific_row, dtype=float, sep=',')
            print("specific_row type: ",type(specific_row))
            specific_rows.append(specific_row)
            print("size: ",size)
            for i in range(size-1):
                
                specific_row = csv_file.readline().strip()
                specific_row = np.fromstring(specific_row, dtype=float, sep=',')
                specific_rows.append(specific_row)
        return np.array(specific_rows)
    #Assign every batch
    def preprocessing(self,xp,assignments):
        clustering_batch = [[] for _ in range(len(self.centroids))]
        for i, k in enumerate(assignments):
            clustering_batch[k].append((xp[i][0],xp[i][1:]))  # the nth vector gets added to the kth cluster...
        return clustering_batch
    # train
    def IVF_train(self):
        xp=self.fetch_from_csv(self.data_path,1,self.train_batch_size)
        # ids = xp[:, 0]
        # get the embed columns
        embeds = xp[:, 1:]
        embeds = np.array([record/np.linalg.norm(record) for record in embeds])
        # xp = xp/np.linalg.norm(xp, axis=1).reshape(-1,1)
        (centroids, assignments) = kmeans2(embeds, self.centroids_num, self.iter,minit='points')
        self.centroids=centroids
        clustering_batch=self.preprocessing(xp,assignments)
        return clustering_batch

    #clustering_data
    def IVF_predict(self):
        xp=self.fetch_from_csv(self.data_path,self.train_batch_size+self.predict_batch_size*self.prediction_count+1,self.predict_batch_size)
        embeds = xp[:, 1:]
        embeds = np.array([record/np.linalg.norm(record) for record in embeds])
        # xp=np.array([record/np.linalg.norm(record) for record in xp])
        self.prediction_count+=1
        assignments, _ = vq(embeds, self.centroids)
        clustering_batch=self.preprocessing(xp,assignments)
        return clustering_batch
    
    #Searching
    def _cal_score(self,vec1, vec2):
        cosine_similarity=vec1.dot(vec2.T).T / (np.linalg.norm(vec1) * np.linalg.norm(vec2))
        return cosine_similarity
    
    def IVF_search(self,query):
        l=[]
        query = query/np.linalg.norm(query)
        for centroid in self.centroids:
            x=self._cal_score(query,centroid)
            x= math.sqrt(2*abs(1-x))
            l.append(x)
        nearset_centers=sorted(range(len(l)), key=lambda sub: l[sub])[:self.nprops]
        return nearset_centers
    
    def IVF_test(self,query,train_batch_clusters):
        l=[]
        query = query/np.linalg.norm(query)
        for centroid in self.centroids:
            x=self._cal_score(query,centroid)
            x= math.sqrt(2*abs(1-x))
            l.append(x)
        nearset_centers=sorted(range(len(l)), key=lambda sub: l[sub])[:self.nprops]
        print("centers len",len(nearset_centers))
        nearest=[]
        for c in nearset_centers:
            for row in train_batch_clusters[c]:
                x=self._cal_score(row[1], query[0])
                x= math.sqrt(2*abs(1-x))
                nearest.append({'index':row[0],'value':x})
        sorted_list = sorted(nearest, key=lambda x: x['value'])[:10]
        return [d['index'] for d in sorted_list]
        

# #Start clustrin
# assignments,centroids=IVF_train(xp,num_part,iter)
# clustering=IVF_clustering(assignments,num_part)

# # # start Evaluation
# # nearset_K_implemented=test_IVF_cosine(query,centroids,xp,clustering,k,nprops);
# # print(nearset_K_implemented)

# # print(len(assignments))

# from typing_extensions import runtime
@dataclass
class Result:
    run_time: float
    top_k: int
    db_ids: List[int]
    actual_ids: List[int]
def run_queries(top_k, num_runs):
    results = []
    for _ in range(num_runs):
        query = np.random.random((1,70))
        query = query/np.linalg.norm(query)
        tic = time.time()
        ivfindex=ivf(data_path="saved_db.csv",train_batch_size=10000,predict_batch_size= 10000,iter=32,centroids_num= 16,nprops=3)
        train_batch_clusters=ivfindex.IVF_train()
        # Clustering
        db_ids=ivfindex.IVF_test(query,train_batch_clusters)
        toc = time.time()
        run_time = toc - tic
        print("time of search:")
        print(run_time)
        print(db_ids)
        np_rows=ivfindex.fetch_from_csv("saved_db.csv",1,10000)
        embeds = np_rows[:, 1:]
        np_rows = np.array([record/np.linalg.norm(record) for record in embeds])
        tic = time.time()
        actual_ids = np.argsort(np_rows.dot(query.T).T / (np.linalg.norm(np_rows, axis=1) * np.linalg.norm(query)), axis= 1).squeeze().tolist()[::-1]
        toc = time.time()
        np_run_time = toc - tic
        print(actual_ids[:30])
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



results=run_queries(10, 10)
# print(results)
print(eval(results))