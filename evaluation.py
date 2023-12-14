import numpy as np
from worst_case_implementation import VecDBWorst
import time
from dataclasses import dataclass
from typing import List
import os
from vec_db import VecDB
from vec_db_ivf import VecDBIVF

# import faiss

AVG_OVERX_ROWS = 10

@dataclass
class Result:
    run_time: float
    top_k: int
    db_ids: List[int]
    actual_ids: List[int]

def run_queries(db, np_rows, top_k, num_runs):
    results = []
    for _ in range(num_runs):
        query = np.random.random((1,70))
        
        tic = time.time()
        db_ids = db.retrive(query, top_k)
        # print("Our indicies ",db_ids)
        # print("db_ids = ",db_ids)
        toc = time.time()
        run_time = toc - tic
        tic = time.time()
        # l2_index = faiss.IndexFlatL2(70)
        # l2_index.add(np_rows)
        # l2_dist, actual_ids = l2_index.search(query,len(np_rows))
        # actual_ids = np.argsort(np_rows.dot(query.T).T / (np.linalg.norm(np_rows, axis=1) * np.linalg.norm(query)), axis= 1).squeeze().tolist()
        actual_ids = np.argsort(np_rows.dot(query.T).T / (np.linalg.norm(np_rows, axis=1) * np.linalg.norm(query)), axis= 1).squeeze().tolist()[::-1]
        # print("actual_ids = ",actual_ids[:5])
        toc = time.time()
        np_run_time = toc - tic
        
        actual_ids = np.ravel(actual_ids).tolist()
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
    return sum(scores) / len(scores), sum(run_time) / len(run_time)


if __name__ == "__main__":

    # try:
    for i in range(1):
        db = VecDBIVF()
        records_np = np.random.random((10000, 70))
        records_dict = [{"id": i, "embed": list(row)} for i, row in enumerate(records_np)]
        _len = len(records_np)
        db.insert_records(records_dict)
        # for i in range(4):
        #   records_np = np.concatenate([records_np, np.random.random((1000000, 70))])
        #   records_dict = [{"id": i + _len, "embed": list(row)} for i, row in enumerate(records_np[_len:])]
        #   _len = len(records_np)
        #   db.insert_records(records_dict)
        # db._build_index()
        res = run_queries(db, records_np, 5, 10)
        print("Evaluation = ",eval(res))
    # except Exception as e:
    #     print("error: ",e)

    # delete codes file
    for i in range(1000000):
        if os.path.exists("codes_"+str(i)+".bin"):
            os.remove("codes_"+str(i)+".bin")
    for i in range(1000000):
        if os.path.exists("ivf_cluster_"+str(i)+".bin"):
            os.remove("ivf_cluster_"+str(i)+".bin")
        else:
            break

 
    # records_np = np.concatenate([records_np, np.random.random((90000, 70))])
    # records_dict = [{"id": i + _len, "embed": list(row)} for i, row in enumerate(records_np[_len:])]
    # _len = len(records_np)
    # db.insert_records(records_dict)
    # print("len = ",len(records_np))
    # res = run_queries(db, records_np, 5, 10)
    # print("Evaluation = ",eval(res))
    
    #     # delete codes file
    # for i in range(1000000):
    #     if os.path.exists("codes_"+str(i)+".txt"):
    #         os.remove("codes_"+str(i)+".txt")
    # for i in range(1000000):
    #     if os.path.exists("ivf_cluster_"+str(i)+".txt"):
    #         os.remove("ivf_cluster_"+str(i)+".txt")
    #     else:
    #         break

    # records_np = np.concatenate([records_np, np.random.random((900000, 70))])
    # records_dict = [{"id": i + _len, "embed": list(row)} for i, row in enumerate(records_np[_len:])]
    # _len = len(records_np)
    # db.insert_records(records_dict)
    # res = run_queries(db, records_np, 5, 10)
    # print("Evaluation = ",eval(res))
    
    # # delete codes file
    # for i in range(1000000):
    #     if os.path.exists("codes_"+str(i)+".txt"):
    #         os.remove("codes_"+str(i)+".txt")
    # for i in range(1000000):
    #     if os.path.exists("ivf_cluster_"+str(i)+".txt"):
    #         os.remove("ivf_cluster_"+str(i)+".txt")
    #     else:
    #         break

    # records_np = np.concatenate([records_np, np.random.random((4000000, 70))])
    # records_dict = [{"id": i + _len, "embed": list(row)} for i, row in enumerate(records_np[_len:])]
    # _len = len(records_np)
    # db.insert_records(records_dict)
    # res = run_queries(db, records_np, 5, 10)
    # eval(res)

    # records_np = np.concatenate([records_np, np.random.random((5000000, 70))])
    # records_dict = [{"id": i + _len, "embed": list(row)} for i, row in enumerate(records_np[_len:])]
    # _len = len(records_np)
    # db.insert_records(records_dict)
    # res = run_queries(db, records_np, 5, 10)
    # eval(res)

    # records_np = np.concatenate([records_np, np.random.random((5000000, 70))])
    # records_dict = [{"id": i +  _len, "embed": list(row)} for i, row in enumerate(records_np[_len:])]
    # _len = len(records_np)
    # db.insert_records(records_dict)
    # res = run_queries(db, records_np, 5, 10)
    # eval(res)

    