
import os

import numpy as np

from catboost.hnsw import HnswEstimator, EDistance


def test_floats_60000():
    data = np.fromfile(os.path.join(os.getcwd(), "hnsw_data", 'floats_60000'), dtype=np.dtype('float32'))
    dimension = 40
    distance = EDistance.DotProduct
    data = data.reshape((data.size // dimension, dimension))
    estimator = HnswEstimator(
        distance=distance,
        level_size_decay=2,
        max_neighbors=5,
        search_neighborhood_size=30,
        batch_size=10,
        upper_level_batch_size=10,
        num_exact_candidates=10
    )
    estimator.fit(data)
    neighbor_distances, neighbor_ids = estimator.kneighbors([data[0]], n_neighbors=20, search_neighborhood_size=40)


if __name__ == '__main__':
    test_floats_60000()
