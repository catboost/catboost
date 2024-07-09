from __future__ import print_function

import pytest
import os

import numpy as np

import yatest.common

import struct

from hnsw import EVectorComponentType, EDistance, Pool, Hnsw, HnswException, transform_mobius, HnswEstimator, OnlineHnsw


np.set_printoptions(legacy='1.13')

DATA_FOLDER = yatest.common.source_path(os.path.join('library', 'python', 'hnsw', 'ut', 'data'))
FLOATS_0 = yatest.common.source_path(os.path.join(DATA_FOLDER, "floats_0"))
FLOATS_1 = yatest.common.source_path(os.path.join(DATA_FOLDER, "floats_1"))
FLOATS_60000 = yatest.common.source_path(os.path.join(DATA_FOLDER, 'floats_60000'))
INT8_100000 = yatest.common.source_path(os.path.join(DATA_FOLDER, 'int8_100000'))
INT32_50000 = yatest.common.source_path(os.path.join(DATA_FOLDER, 'int32_50000'))


POOL_PARAMS = [{'pool_file': FLOATS_60000, 'component_type': EVectorComponentType.Float, 'dtype': np.dtype('float32')},
               {'pool_file': INT8_100000, 'component_type': EVectorComponentType.I8, 'dtype': np.dtype('int8')},
               {'pool_file': INT32_50000, 'component_type': EVectorComponentType.I32, 'dtype': np.dtype('int32')}]
POOL_IDS = ['float', 'int8', 'int32']

DISTANCE_PARAMS = [EDistance.DotProduct, EDistance.L1, EDistance.L2Sqr, EDistance.PairVectorDistance]
DISTACE_IDS = ['DotProduct', 'L1', 'L2Sqr', 'PairVectorDistance']


@pytest.mark.parametrize('pool_params', POOL_PARAMS, ids=POOL_IDS)
def test_pool(pool_params):
    pool = Pool.from_file(pool_params['pool_file'], pool_params['component_type'], 10)
    log_path = yatest.common.test_output_path('log')
    with open(log_path, 'w') as log_file:
        print(pool.get_num_items(), file=log_file)
        print(list(pool.get_item(0)), file=log_file)
    return [yatest.common.canonical_file(log_path, local=True)]


@pytest.mark.parametrize('pool_params', POOL_PARAMS, ids=POOL_IDS)
@pytest.mark.parametrize('distance', DISTANCE_PARAMS, ids=DISTACE_IDS)
def test_index(pool_params, distance):
    pool = Pool.from_file(pool_params['pool_file'], pool_params['component_type'], 10)
    hnsw = Hnsw()
    hnsw.build(pool, distance, level_size_decay=2, max_neighbors=5, search_neighborhood_size=30, batch_size=10,
               upper_level_batch_size=10, num_exact_candidates=10, num_threads=1)
    index_path = yatest.common.test_output_path('index')
    hnsw.save(index_path)
    hnsw.load(index_path, pool, distance)
    neighbors = hnsw.get_nearest(pool.get_item(0), 20, 40)
    log_path = yatest.common.test_output_path('log')
    with open(log_path, 'w') as log_file:
        for neighbor in neighbors:
            print(neighbor, file=log_file)
    return [yatest.common.canonical_file(index_path, local=True),
            yatest.common.canonical_file(log_path, local=True)]


@pytest.mark.parametrize('pool_params', POOL_PARAMS, ids=POOL_IDS)
@pytest.mark.parametrize('distance', DISTANCE_PARAMS, ids=DISTACE_IDS)
def test_save_load(pool_params, distance):
    pool = Pool.from_file(pool_params['pool_file'], pool_params['component_type'], 10)
    hnsw_1 = Hnsw()
    hnsw_1.build(pool, distance, max_neighbors=5, search_neighborhood_size=30, batch_size=10,
                 num_exact_candidates=10, num_threads=1)
    neighbors_before_save = hnsw_1.get_nearest(pool.get_item(0), 20, 40)
    index_path = yatest.common.test_output_path('index')
    hnsw_1.save(index_path)
    hnsw_2 = Hnsw()
    hnsw_2.load(index_path, pool, distance)
    neighbors_after_load = hnsw_2.get_nearest(pool.get_item(0), 20, 40)
    assert neighbors_before_save == neighbors_after_load


def test_work_with_empty_index():
    hnsw = Hnsw()
    with pytest.raises(HnswException):
        hnsw.get_nearest([], 1, 1)
    with pytest.raises(HnswException):
        hnsw.save("index")


def test_index_with_empty_storage():
    pool = Pool.from_file(FLOATS_0, EVectorComponentType.Float, 1)
    hnsw = Hnsw()
    hnsw.build(pool, EDistance.DotProduct)
    assert len(hnsw.get_nearest([0], 10, 10)) == 0


def test_index_with_storage_of_one():
    pool = Pool.from_file(FLOATS_1, EVectorComponentType.Float, 1)
    hnsw = Hnsw()
    hnsw.build(pool, EDistance.DotProduct)
    neighbors = hnsw.get_nearest([0], 10, 10)
    assert len(neighbors) == 1
    assert neighbors[0][0] == 0


def test_load_float_from_bytes():
    EPS = 1e-6
    array = [
        [1.0, 2.0, 3.0],
        [0.0, -1.0, -2.0],
        [111.0, 0.5, 3.141592]
    ]
    vector_bytes = bytes()
    for vector in array:
        for value in vector:
            vector_bytes += struct.pack('f', value)
    pool = Pool.from_bytes(vector_bytes, EVectorComponentType.Float, 3)
    for i in range(pool.get_num_items()):
        for j in range(pool.dimension):
            assert abs(pool.get_item(i)[j] - array[i][j]) < EPS
    assert pool.dimension == 3
    assert pool.get_num_items() == 3


@pytest.mark.parametrize('pool_params', POOL_PARAMS, ids=POOL_IDS)
def test_compare_load(pool_params):
    pool_1 = Pool.from_file(pool_params['pool_file'], pool_params['component_type'], 10)
    f = open(pool_params['pool_file'], "rb")
    array = f.read()
    pool_2 = Pool.from_bytes(array, pool_params['component_type'], 10)
    assert pool_1.get_num_items() == pool_2.get_num_items()
    assert pool_1.dimension == pool_2.dimension
    for i in range(pool_1.get_num_items()):
        for j in range(pool_1.dimension):
            assert pool_1.get_item(i)[j] == pool_2.get_item(i)[j]


@pytest.mark.parametrize('pool_params', POOL_PARAMS, ids=POOL_IDS)
@pytest.mark.parametrize('distance', DISTANCE_PARAMS, ids=DISTACE_IDS)
def test_load_index(pool_params, distance):
    pool = Pool.from_file(pool_params['pool_file'], pool_params['component_type'], 10)
    hnsw_1 = Hnsw()
    hnsw_1.build(pool, distance, max_neighbors=5, search_neighborhood_size=30, batch_size=10,
                 num_exact_candidates=10, num_threads=1)
    index_path = yatest.common.test_output_path('index')
    hnsw_1.save(index_path)
    index_data = open(index_path, 'rb').read()
    neighbors_before_save = hnsw_1.get_nearest(pool.get_item(0), 20, 40)
    hnsw_2 = Hnsw()
    hnsw_2.load_from_bytes(index_data, pool, distance)
    neighbors_after_load = hnsw_2.get_nearest(pool.get_item(0), 20, 40)
    assert neighbors_before_save == neighbors_after_load


def test_mobius_transform_float():
    EPS = 1e-6
    vectors = np.array(
        [
            [0.1, 0.1, 0.1, 0.1],
            [10.0, 10.0, 0.0, 0.0],
            [1.0, 0.0, 0.0, 0.0],
            [-1.0, -1.0, -1.0, -1.0]
        ],
        np.float32
    )
    expected_vectors = [
        [2.5, 2.5, 2.5, 2.5],
        [0.05, 0.05, 0, 0],
        [1, 0, 0, 0],
        [-0.25, -0.25, -0.25, -0.25]
    ]
    vector_bytes = vectors.tobytes()
    pool = Pool.from_bytes(vector_bytes, EVectorComponentType.Float, 4)
    transformed_pool = Pool.from_bytes(bytes(0), EVectorComponentType.Float, 4)
    transformed_pool = transform_mobius(pool)
    for i in range(transformed_pool.get_num_items()):
        for j in range(transformed_pool.dimension):
            assert abs(transformed_pool.get_item(i)[j] - expected_vectors[i][j]) < EPS
            assert abs(pool.get_item(i)[j] - vectors[i][j]) < EPS
    assert transformed_pool.dimension == 4
    assert transformed_pool.get_num_items() == 4
    assert transformed_pool.dtype == EVectorComponentType.Float
    assert pool.dtype == EVectorComponentType.Float


def test_mobius_transform_i8():
    EPS = 1e-6
    vectors = np.array(
        [
            [12, 13, 14, 15],
            [125, 125, 125, 125],
            [1, 0, 0, 0],
            [-1, -1, -1, -1]
        ],
        np.int8
    )
    expected_vectors = [
        [0.0163487738, 0.0177111717, 0.0190735695, 0.0204359673],
        [0.002, 0.002, 0.002, 0.002],
        [1, 0, 0, 0],
        [-0.25, -0.25, -0.25, -0.25]
    ]
    vector_bytes = vectors.tobytes()
    pool = Pool.from_bytes(vector_bytes, EVectorComponentType.I8, 4)
    transformed_pool = transform_mobius(pool)
    for i in range(transformed_pool.get_num_items()):
        for j in range(transformed_pool.dimension):
            assert abs(transformed_pool.get_item(i)[j] - expected_vectors[i][j]) < EPS
            assert pool.get_item(i)[j] == vectors[i][j]
    assert transformed_pool.dimension == 4
    assert transformed_pool.get_num_items() == 4
    assert transformed_pool.dtype == EVectorComponentType.Float
    assert pool.dtype == EVectorComponentType.I8


def test_mobius_transform_i32():
    EPS = 1e-6
    vectors = np.array(
        [
            [12, 13, -14, 15],
            [1000000000, 1000000000, 1000000000, 1000000000],
            [1, 0, 0, 0],
            [-1, -1, -1, -1]
        ],
        np.int32
    )
    expected_vectors = [
        [0.0163487738, 0.0177111717, -0.0190735695, 0.0204359673],
        [1.25e-10, 1.25e-10, 1.25e-10, 1.25e-10],
        [1, 0, 0, 0],
        [-0.25, -0.25, -0.25, -0.25]
    ]
    vector_bytes = vectors.tobytes()
    pool = Pool.from_bytes(vector_bytes, EVectorComponentType.I32, 4)
    transformed_pool = Pool.from_bytes(bytes(0), EVectorComponentType.Float, 4)
    transformed_pool = transform_mobius(pool)
    for i in range(transformed_pool.get_num_items()):
        for j in range(transformed_pool.dimension):
            assert abs(transformed_pool.get_item(i)[j] - expected_vectors[i][j]) < EPS
            assert pool.get_item(i)[j] == vectors[i][j]
    assert transformed_pool.dimension == 4
    assert transformed_pool.get_num_items() == 4
    assert transformed_pool.dtype == EVectorComponentType.Float
    assert pool.dtype == EVectorComponentType.I32


@pytest.mark.parametrize('pool_params', POOL_PARAMS, ids=POOL_IDS)
@pytest.mark.parametrize('distance', DISTANCE_PARAMS, ids=DISTACE_IDS)
@pytest.mark.parametrize('dimension', [10, 20, 40], ids=['dimension=%i' % i for i in [10, 20, 30]])
def test_hnsw_estimator(pool_params, distance, dimension):
    data = np.fromfile(pool_params['pool_file'], dtype=pool_params['dtype'])
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
    neighbor_ids2 = estimator.kneighbors([data[0]], n_neighbors=20, search_neighborhood_size=40, return_distance=False)
    assert np.array_equal(neighbor_ids, neighbor_ids2)

    neighbor_distances_file = yatest.common.test_output_path('neighbor_distances')
    np.savetxt(neighbor_distances_file, neighbor_distances)

    neighbor_ids_file = yatest.common.test_output_path('neighbor_ids')
    np.savetxt(neighbor_ids_file, neighbor_ids, fmt='%i')

    return [yatest.common.canonical_file(neighbor_distances_file, local=True),
            yatest.common.canonical_file(neighbor_ids_file, local=True)]


@pytest.mark.parametrize('pool_params', POOL_PARAMS, ids=POOL_IDS)
def test_online_hnsw_pool(pool_params):
    pool = Pool.from_file(pool_params['pool_file'], pool_params['component_type'], 10)
    online_hnsw = OnlineHnsw(pool_params['component_type'], 10, EDistance.DotProduct)
    for i in range(pool.get_num_items()):
        online_hnsw.add_item(pool.get_item(i))
    assert online_hnsw.get_num_items() == pool.get_num_items()
    for i in range(pool.get_num_items()):
        online_hnsw_item = online_hnsw.get_item(i)
        item = pool.get_item(i)
        assert np.all(online_hnsw_item == item)


@pytest.mark.parametrize('pool_params', POOL_PARAMS, ids=POOL_IDS)
@pytest.mark.parametrize('distance', DISTANCE_PARAMS, ids=DISTACE_IDS)
def test_online_hnsw_index(pool_params, distance):
    pool = Pool.from_file(pool_params['pool_file'], pool_params['component_type'], 10)
    online_hnsw = OnlineHnsw(pool_params['component_type'], 10, distance, level_size_decay=2, max_neighbors=5, search_neighborhood_size=50)
    for i in range(pool.get_num_items()):
        online_hnsw.add_item(pool.get_item(i))
    online_hnsw_1_neighbors = online_hnsw.get_nearest(online_hnsw.get_item(0), 20)
    online_hnsw_2_neighbors = online_hnsw.get_nearest_and_add_item(online_hnsw.get_item(0))
    assert online_hnsw.get_num_items() == pool.get_num_items() + 1
    assert len(online_hnsw_2_neighbors) == 50
    online_hnsw_2_neighbors = online_hnsw_2_neighbors[:20]
    assert online_hnsw_1_neighbors == online_hnsw_2_neighbors
