import numpy as np
from worst_case_implementation import VecDBWorst
import time
from dataclasses import dataclass
from typing import List
import os
from vec_db_ivf import VecDBIVF
from memory_profiler import memory_usage

AVG_OVERX_ROWS = 10

@dataclass
class Result:
    run_time: float
    top_k: int
    db_ids: List[int]
    actual_ids: List[int]
results = []
to_print_arr = []

def run_queries(db, query, top_k, actual_ids, num_runs):
    global results
    results = []
    for _ in range(num_runs):
        tic = time.time()
        db_ids = db.retrive(query, top_k)
        toc = time.time()
        run_time = toc - tic
        results.append(Result(run_time, top_k, db_ids, actual_ids))
    return results

def memory_usage_run_queries(args):
    global results
    # This part is added to calcauate the RAM usage
    mem_before = max(memory_usage())
    mem = memory_usage(proc=(run_queries, args, {}), interval = 1e-3)
    return results, max(mem) - mem_before

def evaluate_result(results: List[Result]):
    # scores are negative. So getting 0 is the best score.
    scores = []
    run_time = []
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
            except:
                score -= len(res.actual_ids)
        scores.append(score)

    return sum(scores) / len(scores), sum(run_time) / len(run_time)


if __name__ == "__main__":
    '''
     ####   changes at each run
     1- cahnges size of each vectores
     2- in vbifv ===>self.file_path 
                 ===>self.data_size
    '''
    
    
    QUERY_SEED_NUMBER = 10
    DB_SEED_NUMBER = 20

    rng = np.random.default_rng(DB_SEED_NUMBER)
    vectors = rng.random((10**5, 70), dtype=np.float32)

    rng = np.random.default_rng(QUERY_SEED_NUMBER)
    query = rng.random((1, 70), dtype=np.float32)

    db = VecDBIVF( new_db = True)
    
    db._build_index()
    
    actual_ids = actual_sorted_ids_10k = np.argsort(vectors.dot(query.T).T / (np.linalg.norm(vectors, axis=1) * np.linalg.norm(query)), axis= 1).squeeze().tolist()[::-1]

    res = run_queries(db, query, 5, actual_ids, 1)  # one run to make everything fresh and loaded
    res, mem = memory_usage_run_queries((db, query, 5, actual_ids, 3)) # actual runs to compute time, and memory
    eval = evaluate_result(res)
    to_print = f"100K\tscore\t{eval[0]}\ttime\t{eval[1]:.2f}\tRAM\t{mem:.2f} MB"
    to_print_arr.append(to_print)
    print(to_print)

    