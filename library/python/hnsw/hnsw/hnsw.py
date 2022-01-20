import sys
import imp
import os
from six import iteritems
from enum import IntEnum
from contextlib import contextmanager
import json


def get_so_paths(dir_name):
    dir_name = os.path.join(os.path.dirname(__file__), dir_name)
    list_dir = os.listdir(dir_name) if os.path.isdir(dir_name) else []
    return [os.path.join(dir_name, so_name) for so_name in list_dir if so_name.split('.')[-1] in ['so', 'pyd']]


def get_hnsw_bin_module():
    if '_hnsw' in sys.modules:
        return sys.modules['_hnsw']
    so_paths = get_so_paths('./')
    for so_path in so_paths:
        try:
            loaded_hnsw = imp.load_dynamic('_hnsw', so_path)
            sys.modules['hnsw._hnsw'] = loaded_hnsw
            return loaded_hnsw
        except ImportError:
            pass
    from . import _hnsw
    return _hnsw


@contextmanager
def log_fixup():
    _hnsw._set_logger(sys.stdout)
    try:
        yield
    finally:
        _hnsw._reset_logger()


class EDistance(IntEnum):
    DotProduct = 0
    L1 = 1
    L2Sqr = 2


class EVectorComponentType(IntEnum):
    Float = 0
    I8 = 1
    I32 = 2

_hnsw = get_hnsw_bin_module()

HnswException = _hnsw.HnswException

_DenseVectorStorage = {
    EVectorComponentType.Float: _hnsw._DenseFloatVectorStorage,
    EVectorComponentType.I8: _hnsw._DenseI8VectorStorage,
    EVectorComponentType.I32: _hnsw._DenseI32VectorStorage
}

_HnswDenseVectorIndex = {
    EVectorComponentType.Float: _hnsw._HnswDenseFloatVectorIndex,
    EVectorComponentType.I8: _hnsw._HnswDenseI8VectorIndex,
    EVectorComponentType.I32: _hnsw._HnswDenseI32VectorIndex
}

_transform_mobius = {
    EVectorComponentType.Float: _hnsw._transform_mobius_float,
    EVectorComponentType.I8: _hnsw._transform_mobius_i8,
    EVectorComponentType.I32: _hnsw._transform_mobius_i32
}

_OnlineHnswDenseVectorIndex = {
    EVectorComponentType.Float: _hnsw._OnlineHnswDenseFloatVectorIndex,
    EVectorComponentType.I8: _hnsw._OnlineHnswDenseI8VectorIndex,
    EVectorComponentType.I32: _hnsw._OnlineHnswDenseI32VectorIndex,
}


class Pool:
    """
    Pool is a storage of vectors
    """

    def __init__(self, vectors_path, dtype, dimension, vectors_bin_data=None):
        """
        Pool is a storage of vectors. You can create it from row-major binary file or
        binary data of vectors.

        Parameters
        ----------
        vectors_path : string
            Path to binary file with vectors.

        dtype : EVectorComponentType
            Type of vectors.

        dimension : int
            Dimension of vectors.

        vectors_bin_data : bytes
            Binary data of vectors.
        """
        self.vectors_path = vectors_path
        self.dtype = dtype
        self.dimension = dimension
        assert (vectors_bin_data is None) ^ (vectors_path is None)
        if vectors_path is not None:
            self._storage = _DenseVectorStorage[dtype](vectors_path, dimension)
            self._data = None
        if vectors_bin_data is not None:
            self._storage = _DenseVectorStorage[dtype](None, dimension, vectors_bin_data)
            self._data = vectors_bin_data

    @classmethod
    def from_file(cls, vectors_path, dtype, dimension):
        """
        Create pool from binary file.

        Parameters
        ----------
        vectors_path : string
            Path to binary file with vectors.

        dtype : EVectorComponentType
            Type of vectors.

        dimension : int
            Dimension of vectors.
        """
        return Pool(vectors_path, dtype, dimension, None)

    @classmethod
    def from_bytes(cls, vectors_bin_data, dtype, dimension):
        """
        Create pool from binary data.

        Parameters
        ----------
        vectors_bin_data : bytes
            Binary data of vectors.

        dtype : EVectorComponentType
            Type of vectors.

        dimension : int
            Dimension of vectors.
        """
        return Pool(None, dtype, dimension, vectors_bin_data)

    def get_item(self, id):
        """
        Get item from storage by id.

        Parameters
        ----------
        id : int
            Index of item in storage.

        Returns
        -------
        item : numpy.ndarray
        """
        return self._storage._get_item(id)

    def get_num_items(self):
        """
        Get the number of items in storage.

        Returns
        -------
        num_items : int
        """
        return self._storage._get_num_items()


def transform_mobius(pool):
    """
    Transform pool for fast dot product search on HNSW graph
    https://papers.nips.cc/paper/9032-mobius-transformation-for-fast-inner-product-search-on-graph.pdf

    Parameters
    ----------
    pool : Pool

    Returns
    -------
    transformed_pool : Pool
    """
    transformed_pool = Pool.from_bytes(bytes(0), EVectorComponentType.Float, pool.dimension)
    transformed_pool._storage = _transform_mobius[pool.dtype](pool._storage)
    return transformed_pool


class Hnsw:
    """
    Class for building, loading and working with Hierarchical Navigable Small World index.
    """
    def __init__(self):
        """
        Create object for working with HNSW.
        """
        self._index = None
        self._data = None

    def build(self, pool, distance, max_neighbors=None, search_neighborhood_size=None, num_exact_candidates=None,
              batch_size=None, upper_level_batch_size=None, level_size_decay=None, num_threads=None, verbose=False,
              report_progress=True, snapshot_file=None, snapshot_interval=None):
        """
        Build index with given options.

        Parameters
        ----------
        pool : Pool
            Pool of vectors for which index will be built.

        distance : EDistance
            Distance that should be used for finding nearest vectors.

        max_neighbors : int (default=32)
            Maximum number of neighbors that every item can be connected with.

        search_neighborhood_size : int (default=300)
            Search neighborhood size for ANN-search.
            Higher values improve search quality in expense of building time.

        num_exact_candidates : int (default=100)
            Number of nearest vectors to take from batch.
            Higher values improve search quality in expense of building time.

        batch_size : int (default=1000)
            Number of items that added to graph on each step of algorithm.

        upper_level_batch_size : int (default=40000)
            Batch size for building upper levels.

        level_size_decay : int (default=max_neighbors/2)
            Base of exponent for decaying level sizes.

        num_threads : int (default=number of CPUs)
            Number of threads for building index.

        report_progress : bool (default=True)
            Print progress of building.

        verbose : bool (default=False)
            Print additional information about time of building.

        snapshot_file : string (default=None)
            Path for saving snapshots during the index building.

        snapshot_interval : int (default=600)
            Interval between saving snapshots (seconds).
            Snapshot is saved after building each level also.
        """
        params = {}
        not_params = ["not_params", "self", "params", "__class__", "pool", "distance"]
        for key, value in iteritems(locals()):
            if key not in not_params and value is not None:
                params[key] = value
        self._index = _HnswDenseVectorIndex[pool.dtype](pool._storage, distance)
        with log_fixup():
            self._index._build(json.dumps(params))

    def _check_index(self):
        if self._index is None:
            raise HnswException("Index is not built and not loaded")

    def save(self, index_path):
        """
        Save index to file.

        Parameters
        ----------
        index_path : string
            Path to file for saving index.
        """
        self._check_index()
        self._index._save(index_path)

    def load(self, index_path, pool, distance):
        """
        Load index from file.

        Parameters
        ----------
        index_path : string
            Path to file for loading index.

        pool : Pool
            Pool of vectors for which index will be loaded.

        distance : EDistance
            Distance that should be used for finding nearest vectors.
        """
        self._index = _HnswDenseVectorIndex[pool.dtype](pool._storage, distance)
        self._index._load(index_path)
        self._data = None

    def load_from_bytes(self, index_data, pool, distance):
        """
        Load index from bytes.

        Parameters
        ----------
        index_data : bytes
            Index binary data.

        pool : Pool
            Pool of vectors for which index will be loaded.

        distance : EDistance
            Distance that should be used for finding nearest vectors.
        """
        self._index = _HnswDenseVectorIndex[pool.dtype](pool._storage, distance)
        self._index._load_from_bytes(index_data)
        self._data = index_data

    def get_nearest(self, query, top_size, search_neighborhood_size, distance_calc_limit=0):
        """
        Get approximate nearest neighbors for query from index.

        Parameters
        ----------
        query : list or numpy.ndarray
            Vector for which nearest neighbors should be found.

        top_size : int
            Required number of neighbors.

        search_neighborhood_size : int
            Search neighborhood size for ANN-search.
            Higher values improve search quality in expense of search time.
            It should be equal or greater than top_size.

        distance_calc_limit : int (default=0)
            Limit of distance calculation.
            To guarantee satisfactory search time at the expense of quality.
            0 is equivalent to no limit.

        Returns
        -------
        neighbors : list of tuples (id, distance)
        """
        self._check_index()
        return self._index._get_nearest(query, top_size, search_neighborhood_size, distance_calc_limit)


class HnswEstimator:
    """
    Class for building, loading and working with Hierarchical Navigable Small World index with SciKit-Learn
    Estimator compatible interface.
    Mostly drop-in replacement for sklearn.neighbors.NearestNeighbors (except for some parameters)
    """

    def __init__(self, n_neighbors=5,
                 distance=EDistance.DotProduct, max_neighbors=32, search_neighborhood_size=300,
                 num_exact_candidates=100, batch_size=1000, upper_level_batch_size=40000,
                 level_size_decay=None):
        """
        Parameters
        ----------
        n_neighbors : int, default=5
            Number of neighbors to use by default for kneighbors queries.


        distance : EDistance
            Distance that should be used for finding nearest vectors.

        max_neighbors : int (default=32)
            Maximum number of neighbors that every item can be connected with.

        search_neighborhood_size : int (default=300)
            Search neighborhood size for ANN-search.
            Higher values improve search quality in expense of building time.

        num_exact_candidates : int (default=100)
            Number of nearest vectors to take from batch.
            Higher values improve search quality in expense of building time.

        batch_size : int (default=1000)
            Number of items that added to graph on each step of algorithm.

        upper_level_batch_size : int (default=40000)
            Batch size for building upper levels.

        level_size_decay : int (default=max_neighbors/2)
            Base of exponent for decaying level sizes.
        """
        for key, value in iteritems(locals()):
            if key not in ['self', '__class__']:
                setattr(self, key, value)

    def _check_index(self):
        if self._index is None:
            raise HnswException("Index is not built and not loaded")

    def fit(self, X, y=None, num_threads=None, verbose=False, report_progress=True, snapshot_file=None,
            snapshot_interval=600):
        """
        Fit the HNSW model.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_values)

        y: None
            Added to be compatible with Estimator API

        num_threads : int (default=number of CPUs)
            Number of threads for building index.

        report_progress : bool (default=True)
            Print progress of building.

        verbose : bool (default=False)
            Print additional information about time of building.

        snapshot_file : string (default=None)
            Path for saving snapshots during the index building.

        snapshot_interval : int (default=600)
            Interval between saving snapshots (seconds).

        Returns
        -------
        model : HnswEstimator

        """
        self._index, self._index_data = _hnsw._init_index(X, self.distance)

        params = self._get_params(return_none=False)
        not_params = ["not_params", "self", "params", "__class__", "X", "y"]
        for key, value in iteritems(locals()):
            if key not in not_params and value is not None:
                params[key] = value
        del params['distance']

        with log_fixup():
            self._index._build(json.dumps(params))
        return self

    def _get_params(self, return_none):
        params = {}
        for key, value in self.__dict__.items():
            if key[0] != '_' and (return_none or (value is not None)):
                params[key] = value
        return params

    def get_params(self, deep=True):
        """
        Get parameters for this estimator.
        """
        return self._get_params(return_none=True)

    def set_params(self, **params):
        """
        Set the parameters of this estimator.

        Parameters
        ----------
        **params : dict
            HnswEstimator parameters.

        Returns
        -------
        self : HnswEstimator instance
        """
        if not params:
            return self
        valid_params = self._get_params(return_none=True)

        for key, value in params.items():
            if key not in valid_params:
                raise HnswException(
                    'Invalid parameter %s for HnswEstimator. '
                    'Check the list of available parameters '
                    'with `get_params().keys()`.'
                )
                setattr(self, key, value)

        return self

    @property
    def effective_metric_(self):
        """
        Returns
        -------
        Distance that should be used for finding nearest vectors.
        """
        return self.distance

    @property
    def n_samples_fit_(self):
        """
        Returns
        -------
        Number of samples in the fitted data.
        """
        self._check_index()
        return self._index_data.shape[0]

    def kneighbors(self, X=None, n_neighbors=None, return_distance=True, search_neighborhood_size=None,
                   distance_calc_limit=0):
        """Finds the approximate K-neighbors of a point.
        Returns indices of and distances to the neighbors of each point.

        Parameters
        ----------
        X : array-like, shape (n_queries, n_features) or None
            The query point or points.
            If not provided, neighbors of each indexed point are returned.
            In this case, the query point is not considered its own neighbor.
        n_neighbors : int, default=None
            Number of neighbors required for each sample. The default is the
            value passed to the constructor.
        return_distance : bool, default=True
            Whether or not to return the distances.

        search_neighborhood_size : int, default=None
            Search neighborhood size for ANN-search.
            Higher values improve search quality in expense of search time.
            It should be equal or greater than top_size.
            If None set to n_neighbors * 2.

        distance_calc_limit : int (default=0)
            Limit of distance calculation.
            To guarantee satisfactory search time at the expense of quality.
            0 is equivalent to no limit.

        Returns
        -------
        neigh_dist :numpy.ndarray of shape (n_queries, n_neighbors)
            Array representing the lengths to points, only present if
            return_distance=True
        neigh_ind : numpy.ndarray of shape (n_queries, n_neighbors)
            Indices of the nearest points in the population matrix.
        """
        self._check_index()

        if X is None:
            X = self._index_data
        if n_neighbors is None:
            n_neighbors = self.n_neighbors
        if search_neighborhood_size is None:
            search_neighborhood_size = n_neighbors * 2

        return self._index._kneighbors(X, n_neighbors, return_distance, self.distance, search_neighborhood_size,
                                       distance_calc_limit)


class OnlineHnsw:
    """
    Class for building and working with Online Hierarchical Navigable Small World index.
    """
    def __init__(self, dtype, dimension, distance, max_neighbors=None, search_neighborhood_size=None, num_vertices=None, level_size_decay=None):
        """
        Create object with given options.

        Parameters
        ----------
        dtype : EVectorComponentType
            Type of vectors.
        dimension : int
            Dimension of vectors.
        distance : EDistance
            Distance that should be used for finding nearest vectors.
        max_neighbors : int (default=32)
            Maximum number of neighbors that every item can be connected with.
        search_neighborhood_size : int (default=300)
            Search neighborhood size for ANN-search.
            Higher values improve search quality in expense of building time.
        num_vertices : int (default=0)
            Expected number of vectors in storage.
        level_size_decay : int (default=max_neighbors/2)
            Base of exponent for decaying level sizes.
        """
        self.dtype = dtype
        self.dimension = dimension
        params = {}
        all_params = ["max_neighbors", "search_neighborhood_size", "num_vertices", "level_size_decay"]
        for key, value in iteritems(locals()):
            if key in all_params and value is not None:
                params[key] = value
        self._online_index = _OnlineHnswDenseVectorIndex[dtype](dimension, distance, json.dumps(params))

    def get_nearest_and_add_item(self, query):
        """
        Get approximate nearest neighbors for query from index and add item to index

        Parameters
        ----------
        query : list or numpy.ndarray
            Vector for which nearest neighbors should be found.
            Vector which should be added in index.

        Returns
        -------
        neighbors : list of tuples (id, distance) with length = search_neighborhood_size
        """
        return self._online_index._get_nearest_neighbors_and_add_item(query)

    def get_nearest(self, query, top_size=0):
        """
        Get approximate nearest neighbors for query from index.

        Parameters
        ----------
        query : list or numpy.ndarray
            Vector for which nearest neighbors should be found.
        top_size : int
            Required number of neighbors.

        Returns
        -------
        neighbors : list of tuples (id, distance)
        """
        return self._online_index._get_nearest_neighbors(query, top_size)

    def add_item(self, item):
        """
        Add item in index.

        Parameters
        ----------
        item : list or numpy.ndarray
            Vector which should be added in index.
        """
        self._online_index._add_item(item)

    def get_item(self, id):
        """
        Get item from storage by id.

        Parameters
        ----------
        id : int
            Index of item in storage.

        Returns
        -------
        item : numpy.ndarray
        """
        return self._online_index._get_item(id)

    def get_num_items(self):
        """
        Get the number of items in storage.

        Returns
        -------
        num_items : int
        """
        return self._online_index._get_num_items()
