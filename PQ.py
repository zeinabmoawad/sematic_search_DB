from __future__ import annotations

import logging

import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics.pairwise import euclidean_distances
import pickle
import os
import sklearn.metrics.pairwise as pw


logger = logging.getLogger(__name__)

BITS2DTYPE = {
    8: np.uint8,
    4: np.uint8,
    1: np.uint8,
    2: np.uint8,
    3: np.uint8,
    5: np.uint16,
    6: np.uint16
}


def save (file_path,estimators):
    with open(file_path, 'wb') as file:
        pickle.dump(estimators, file)

def load(file_path):
    with open(file_path, 'rb') as file:
        estimators = pickle.load(file)
    return estimators


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

        self.estimators = [
            KMeans(n_clusters=self.k, **estimator_kwargs) for _ in range(m)
        ]
        logger.info(f"Creating following estimators: {self.estimators[0]!r}")
        save (self.estimator_file,self.estimators) #"estimators.pkl"

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
        # print the numpy array
        # print("embeddings",embeds)
        # print("ids",ids)
        # print("shape of embeddings",embeds.shape)
        # print("shape of ids",ids.shape)
        # print("type of embeddings",embeds.dtype)
        # print("type of ids",ids.dtype)
        return embeds,ids
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
        X,ids = self.load_db()
        print("start training")
        for i in range(self.m):
            estimator = self.estimators[i]
            X_i = X[:, i * self.ds : (i + 1) * self.ds]

            logger.info(f"Fitting KMeans for the {i}-th segment")
            estimator.fit(X_i)
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
        with open(self.path_to_db, "r") as fin:
            for row in fin.readlines():
                row_splits = row.split(",")
                id = int(float(row_splits[0]))
                # print("id = ", np.array([id]))
                X = [float(e) for e in row_splits[1:]]
                X = np.array(X).reshape(1,self.d)
                code = np.empty((1, self.m), dtype=self.dtype)
                for i in range(self.m):
                    estimator = self.estimators[i]
                    X_i = X[:, i * self.ds : (i + 1) * self.ds]
                    code[:, i] = estimator.predict(X_i)
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

    def add(self) -> None:
        """Add vectors to the database (their encoded versions).

        Parameters
        ----------
        X
            Array of shape `(n_codes, d)` of dtype `np.float32`.
        """
        if not self.is_trained:
            raise ValueError("The quantizer needs to be trained first.")
        # self.codes = self.encode()
        save (self.codes_file,self.encode()) #"codes.pkl"

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
            centers = self.estimators[i].cluster_centers_  # (k, ds)
            distance_table[:, i, :] = euclidean_distances(
                X_i, centers, squared=True
            )

        distances = np.zeros((n_queries, n_codes), dtype=self.dtype_orig)

        for i in range(self.m):
            distances += distance_table[:, i, self.codes[:, i]]

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
        
        
        indices = np.argsort(distances_all, axis=1)[:, :k]

        # distances = np.empty((n_queries, k), dtype=np.float32)
        # for i in range(n_queries):
            # distances[i] = distances_all[i][indices[i]]
        return indices[0]
        # return np.array(distances_all[:k,0]).astype(np.int32)

