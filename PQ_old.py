from __future__ import annotations

import logging

import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics.pairwise import euclidean_distances
import matplotlib.pyplot as plt

logger = logging.getLogger(__name__)

BITS2DTYPE = {
    8: np.uint8,
    4: np.uint8,
    16: np.uint16,
    9: np.uint16,
    2: np.uint8,
    1: np.uint8,
    32: np.uint32,
    3: np.uint8,
}


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
        **estimator_kwargs: str | int,
    ) -> None:
        if d % m != 0:
            raise ValueError("d needs to be a multiple of m")

        if nbits not in BITS2DTYPE:
            raise ValueError(f"Unsupported number of bits {nbits}")

        self.m = m
        self.k = 2**nbits
        self.d = d
        self.ds = d // m

        self.estimators = [
            KMeans(n_clusters=self.k, **estimator_kwargs) for _ in range(m)
        ]
        logger.info(f"Creating following estimators: {self.estimators[0]!r}")

        self.is_trained = False

        self.dtype = BITS2DTYPE[nbits]
        self.dtype_orig = np.float32

        self.codes: np.ndarray | None = None

    def train(self, X: np.ndarray) -> None:
        """Train all KMeans estimators.

        Parameters
        ----------
        X
            Array of shape `(n, d)` and dtype `float32`.

        """
        if self.is_trained:
            raise ValueError("Training multiple times is not allowed")

        for i in range(self.m):
            estimator = self.estimators[i]
            X_i = X[:, i * self.ds : (i + 1) * self.ds]

            logger.info(f"Fitting KMeans for the {i}-th segment")
            estimator.fit(X_i)
            self.estimators[i] = estimator
            # call centersDistance to get distance matrix
            self.distance_matrix = self.centersDistance(estimator.cluster_centers_)

        self.is_trained = True

    def centersDistance(self, centers: np.ndarray) -> np.ndarray:
        # create a distance matrix between centers
        # centers: (k, ds)
        # distance_matrix: (k, k)
        distance_matrix = np.zeros((self.k, self.k), dtype=self.dtype_orig)
        for i in range(self.k):
            for j in range(self.k):
                distance_matrix[i][j] = np.linalg.norm(centers[i] - centers[j])
        return distance_matrix


    def encode(self, X: np.ndarray) -> np.ndarray:
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
        n = len(X)
        result = np.empty((n, self.m), dtype=self.dtype)

        for i in range(self.m):
            estimator = self.estimators[i]
            X_i = X[:, i * self.ds : (i + 1) * self.ds]
            result[:, i] = estimator.predict(X_i)

            # print("centers: ",estimator.cluster_centers_)
            # print("labels: ",np.unique(estimator.labels_))
            # # print("feature_names_in_: ",estimator.feature_names_in_)
            # # plot X_i and estimator.cluster_centers_
            # plt.scatter(X_i[:,0],X_i[:,1])
            # plt.scatter(estimator.cluster_centers_[:,0],estimator.cluster_centers_[:,1])
            # plt.show()

        return result

    def add(self, X: np.ndarray) -> None:
        """Add vectors to the database (their encoded versions).

        Parameters
        ----------
        X
            Array of shape `(n_codes, d)` of dtype `np.float32`.
        """
        if not self.is_trained:
            raise ValueError("The quantizer needs to be trained first.")
        self.codes = self.encode(X)

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
            # measure the distance between X_i and centers and select the minimum distance
            
            distance_table[:, i, :] = euclidean_distances(
                X_i, centers, squared=True
            )
        # get nearest center for each segment for each query
        nearest_centers = np.argmin(distance_table, axis=2)

        # compute the distance between query and codes by using distance_matrix
        # distance_matrix: (k, k)
        # nearest_centers: (n_queries, m)
        # distance_table: (n_queries, m, k)

        # distances = np.zeros((n_queries, n_codes), dtype=self.dtype_orig)
        # for i in range(n_queries):
        #     for k in range(n_codes):
        #         distance = 0
        #         for j in range(self.m):
        #             distance += self.distance_matrix[nearest_centers[i][j]][self.codes[i][j]]
        #         distances[i][k] = distance


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
        n_queries = len(X)
        distances_all = self.compute_asymmetric_distances(X)

        indices = np.argsort(distances_all, axis=1)[:, :k]

        distances = np.empty((n_queries, k), dtype=np.float32)
        for i in range(n_queries):
            distances[i] = distances_all[i][indices[i]]
        # save distances and indices to files in same text file
        np.savetxt("distances_indices.txt", np.concatenate((distances, indices), axis=1), delimiter=",", fmt="%f")
        # convert indices to 1 dimension
        indices = np.ravel(indices)
        # print("indices = ",indices)
        return  indices
    