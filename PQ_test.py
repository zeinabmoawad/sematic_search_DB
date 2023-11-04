import numpy as np
import time
from dataclasses import dataclass
from typing import List
import faiss
from PQ_old import CustomIndexPQ

@dataclass
class Result:
    run_time: float
    top_k: int
    db_ids: List[int]
    actual_ids: List[int]

def run_queries(index, np_rows, top_k, num_runs):
    results = []
    for _ in range(num_runs):
        query = np.random.random((1,70))
        query = np_rows[10]
        # reshape query to (1,70)
        query = query.reshape(1,70)
        tic = time.time()
        db_ids = index.search(query, top_k)
        # print("db_ids = ",db_ids)
        toc = time.time()
        run_time = toc - tic
        tic = time.time()
        l2_index = faiss.IndexFlatL2(70)
        l2_index.add(np_rows)
        l2_dist, actual_ids = l2_index.search(query,len(np_rows))
        # actual_ids = np.argsort(np_rows.dot(query.T).T / (np.linalg.norm(np_rows, axis=1) * np.linalg.norm(query)), axis= 1).squeeze().tolist()
        toc = time.time()
        np_run_time = toc - tic
        actual_ids = np.ravel(actual_ids).tolist()
        results.append(Result(run_time, top_k, db_ids, actual_ids))
    return results

def eval_faiss(data,k):

    query = np.random.random((1,70))
    results = index.search(query, k)
    l2_index = faiss.IndexFlatL2(70)
    l2_index.add(data)
    l2_dist, l2_I = l2_index.search(query,k*3)

    print("success: ",sum([1 for i in results if i in l2_I]))

def eval(results: List[Result]):
    # scores are negative. So getting 0 is the best score.
    scores = []
    run_time = []
    counter = 0
    for res in results:
        run_time.append(res.run_time)
        # case for retireving number not equal to top_k, socre will be the lowest

        # if len(set(res.db_ids)) != res.top_k or len(res.db_ids) != res.top_k:
        #     scores.append( -1 * len(res.actual_ids) * res.top_k)
        #     continue
        score = 0
        for id in res.db_ids:
            try:
                ind = res.actual_ids.index(id)
                if ind > res.top_k * 10:
                    score -= ind
                else :
                    counter += 1
                    # print("*************found*************************************")
            except:
                score -= len(res.actual_ids)
        scores.append(score)
    print(counter)
    return sum(scores) / len(scores), sum(run_time) / len(run_time)

if __name__ == "__main__":

    records_np = np.random.random((1000, 70))
    records_dict = [{"id": i, "embed": list(row)} for i, row in enumerate(records_np)]
    _len = len(records_np)
    index = CustomIndexPQ(m=7,d=70,nbits=8,max_iter=100)
    index.train(records_np)
    index.add(records_np)
    res = run_queries(index, records_np, 5, 10)
    # eval_faiss(records_np,10)
    print("Evaluation = ",eval(res))

    
    # records_np = np.concatenate([records_np, np.random.random((90000, 70))])
    # records_dict = [{"id": i + _len, "embed": list(row)} for i, row in enumerate(records_np[_len:])]
    # _len = len(records_np)
    # index = CustomIndexPQ(m=10,d=70,nbits=8)
    # index.train(records_np)
    # index.add(records_np)
    # res = run_queries(index, records_np, 10, 10)
    # print(eval(res))