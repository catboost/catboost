# distutils: language = c++
# coding: utf-8
# cython: wraparound=False, boundscheck=False
# cython: language_level=2

from cpython.ref cimport PyObject, Py_DECREF
from cpython.version cimport PY_MAJOR_VERSION

from six import PY3

import numpy as np
cimport numpy as np  # noqa

from libcpp cimport bool as bool_t
from libcpp cimport nullptr

from util.generic.string cimport TString
from util.memory.blob cimport TBlob
from util.system.types cimport i8, i32, ui32


np.import_array()


class HnswException(Exception):
    pass


cdef extern from "library/python/hnsw/hnsw/helpers.h" namespace "NHnsw::PythonHelpers":
    void SetPythonInterruptHandler() nogil
    void ResetPythonInterruptHandler() nogil


cdef extern from "library/cpp/hnsw/logging/logging.h" namespace "NHnsw":
    cdef void SetCustomLoggingFunction(void(*func)(const char*, size_t len) except * with gil)
    cdef void RestoreOriginalLogger()


cdef extern from "library/cpp/hnsw/index_builder/dense_vector_storage.h" namespace "NHnsw" nogil:
    cdef cppclass TDenseVectorStorage[T]:
        TDenseVectorStorage(const TString& vectors_filename, size_t dimension) except +
        TDenseVectorStorage(const TBlob& vector_data, size_t dimension) except +
        const T* GetItem(size_t id) except +
        size_t GetNumItems()


cdef extern from "library/python/hnsw/hnsw/helpers.h" namespace "NHnsw::PythonHelpers" nogil:
    cdef TDenseVectorStorage[float]* PyTransformMobius[T](const TDenseVectorStorage[T]* storage)


cdef extern from "library/cpp/hnsw/index/index_base.h" namespace "NHnsw" nogil:
    cdef cppclass THnswIndexBase:
        THnswIndexBase(const TBlob& indexBlob) except +


cdef extern from "library/python/hnsw/hnsw/helpers.h" namespace "NHnsw::PythonHelpers" nogil:
    cdef enum EDistance:
        pass

    PyObject* GetDistanceResultType[T](EDistance distance)

    PyObject* GetNearestNeighbors[T](const THnswIndexBase* index, const T* query, size_t topSize,
                                     size_t searchNeighborhoodSize, size_t distanceCalcLimit,
                                     const TDenseVectorStorage* storage, EDistance distance) except +
    void KNeighbors[T](const THnswIndexBase* index,
                       const T* queries,  # [nQueries x dimension] array
                       size_t nQueries,
                       size_t topSize,
                       size_t searchNeighborhoodSize,
                       size_t distanceCalcLimit,
                       const TDenseVectorStorage* storage,
                       EDistance distance,
                       ui32* resultNeighInd,  # [nQueries x topSize] array

                       # [nQueries x topSize] array, can be null (do not return distance in this case)
                       void* resultNeighDist) except +

    TBlob BuildDenseVectorIndex[T](const TString& jsonOptions, const TDenseVectorStorage[T]* storage,
                                             EDistance distance) except + nogil
    void SaveIndex(const TBlob& indexBlob, const TString& path) except +
    TBlob LoadIndex(const TString &path) except +


cdef extern from "library/python/hnsw/hnsw/helpers.h" namespace "NOnlineHnsw::PythonHelpers" nogil:
    cdef cppclass PyOnlineHnswDenseVectorIndex[T]:
        PyOnlineHnswDenseVectorIndex(const TString& jsonOptions, size_t dimension, EDistance distance) except +
        const T* GetItem(size_t id) except +
        void AddItem(const T* item) except +
        PyObject* GetNearestNeighbors(const T* query, size_t topSize) except +
        PyObject* GetNearestNeighborsAndAddItem(const T* query) except +
        size_t GetNumItems()

cdef extern from "Python.h":
    char* PyUnicode_AsUTF8AndSize(object s, Py_ssize_t* l)

cdef _npbytes_ = np.bytes_
cdef _npunicode_ = np.str_ if np.lib.NumpyVersion(np.__version__) >= '2.0.0' else np.unicode_

cdef to_native_str(binary):
    if PY3:
        return binary.decode()
    return binary

cdef inline TString to_arcadia_string(s) except *:
    cdef const unsigned char[:] bytes_s
    cdef const char* utf8_str_pointer
    cdef Py_ssize_t utf8_str_size
    cdef type s_type = type(s)
    if len(s) == 0:
        return TString()
    if s_type is unicode or s_type is _npunicode_:
        # Fast path for most common case(s).
        if PY_MAJOR_VERSION >= 3:
            # we fallback to calling .encode method to properly report error
            utf8_str_pointer = PyUnicode_AsUTF8AndSize(s, &utf8_str_size)
            if utf8_str_pointer != nullptr:
                return TString(utf8_str_pointer, utf8_str_size)
        else:
            tmp = (<unicode>s).encode('utf8')
            return TString(<const char*>tmp, len(tmp))
    elif s_type is bytes or s_type is _npbytes_:
        return TString(<const char*>s, len(s))

    if PY_MAJOR_VERSION >= 3 and hasattr(s, 'encode'):
        # encode to the specific encoding used inside of the module
        bytes_s = s.encode('utf8')
    else:
        bytes_s = s
    return TString(<const char*>&bytes_s[0], len(bytes_s))


log_cout = None


cdef void _CoutLogPrinter(const char* str, size_t len) except * with gil:
    cdef bytes bytes_str = str[:len]
    log_cout.write(to_native_str(bytes_str))
    log_cout.flush()

cdef object _get_nearest_neighbors_float(const THnswIndexBase* index, const float* query, size_t topSize,
                                        size_t searchNeighborhoodSize, size_t distanceCalcLimit,
                                        const TDenseVectorStorage[float]* storage, EDistance distance):

    cdef PyObject* neighbors
    with nogil:
        neighbors = GetNearestNeighbors[float](index, query, topSize, searchNeighborhoodSize, distanceCalcLimit, storage, distance)
    py_neighbors = <object>neighbors
    Py_DECREF(py_neighbors)
    return py_neighbors

cdef object _get_nearest_neighbors_i8(const THnswIndexBase* index, const i8* query, size_t topSize,
                                        size_t searchNeighborhoodSize, size_t distanceCalcLimit,
                                        const TDenseVectorStorage[i8]* storage, EDistance distance):

    cdef PyObject* neighbors
    with nogil:
        neighbors = GetNearestNeighbors[i8](index, query, topSize, searchNeighborhoodSize, distanceCalcLimit, storage, distance)
    py_neighbors = <object>neighbors
    Py_DECREF(py_neighbors)
    return py_neighbors

cdef object _get_nearest_neighbors_i32(const THnswIndexBase* index, const i32* query, size_t topSize,
                                        size_t searchNeighborhoodSize, size_t distanceCalcLimit,
                                        const TDenseVectorStorage[i32]* storage, EDistance distance):

    cdef PyObject* neighbors
    with nogil:
        neighbors = GetNearestNeighbors[i32](index, query, topSize, searchNeighborhoodSize, distanceCalcLimit, storage, distance)
    py_neighbors = <object>neighbors
    Py_DECREF(py_neighbors)
    return py_neighbors

cpdef _set_logger(cout):
    global log_cout
    log_cout = cout
    SetCustomLoggingFunction(&_CoutLogPrinter)

cpdef _reset_logger():
    RestoreOriginalLogger()


cpdef _transform_mobius_float(_DenseFloatVectorStorage storage):
    cdef _DenseFloatVectorStorage transformed_storage
    transformed_storage = _DenseFloatVectorStorage(None, storage._dimension, bytes(0))
    transformed_storage._storage_impl = PyTransformMobius[float](storage._storage_impl)
    return transformed_storage


cpdef _transform_mobius_i8(_DenseI8VectorStorage storage):
    cdef _DenseFloatVectorStorage transformed_storage
    transformed_storage = _DenseFloatVectorStorage(None, storage._dimension, bytes(0))
    transformed_storage._storage_impl = PyTransformMobius[i8](storage._storage_impl)
    return transformed_storage


cpdef _transform_mobius_i32(_DenseI32VectorStorage storage):
    cdef _DenseFloatVectorStorage transformed_storage
    transformed_storage = _DenseFloatVectorStorage(None, storage._dimension, bytes(0))
    transformed_storage._storage_impl = PyTransformMobius[i32](storage._storage_impl)
    return transformed_storage


cdef class _DenseFloatVectorStorage:
    cdef TDenseVectorStorage[float]* _storage_impl
    cdef size_t _dimension

    def __cinit__(self, vectors_filename = None, dimension = 0, bin_data = None, array_data = None):
        self._dimension = dimension
        if vectors_filename is not None:
            self._load_from_file(to_arcadia_string(vectors_filename))
        if bin_data is not None:
            self._load_from_bytearray(bin_data)
        if array_data is not None:
            self._load_from_array(array_data)

    cdef _load_from_array(self, np.float32_t[:, ::1] data):
        self._storage_impl = new TDenseVectorStorage[float](TBlob.NoCopy(<const char*> &data[0,0], data.size * sizeof(np.float32_t)), self._dimension)

    cdef _load_from_bytearray(self, data):
        self._storage_impl = new TDenseVectorStorage[float](TBlob.NoCopy(<const char*> data, len(data)), self._dimension)

    cdef _load_from_file(self, TString vectors_filename):
        self._storage_impl = new TDenseVectorStorage[float](TString(vectors_filename), self._dimension)

    def __dealloc__(self):
        del self._storage_impl

    def _get_item(self, id):
        return np.array(<float[:self._dimension]>self._storage_impl.GetItem(id))

    def _get_num_items(self):
        return self._storage_impl.GetNumItems()


cdef class _DenseI8VectorStorage:
    cdef TDenseVectorStorage[i8]* _storage_impl
    cdef size_t _dimension

    def __cinit__(self, vectors_filename = None, dimension = 0, bin_data = None, array_data = None):
        self._dimension = dimension
        if vectors_filename is not None:
            self._load_from_file(to_arcadia_string(vectors_filename))
        if bin_data is not None:
            self._load_from_bytearray(bin_data)
        if array_data is not None:
            self._load_from_array(array_data)

    cdef _load_from_array(self, np.int8_t[:, ::1] data):
        self._storage_impl = new TDenseVectorStorage[i8](TBlob.NoCopy(<const char*> &data[0,0], data.size * sizeof(np.int8_t)), self._dimension)

    cdef _load_from_bytearray(self, data):
        self._storage_impl = new TDenseVectorStorage[i8](TBlob.NoCopy(<const char*> data, len(data)), self._dimension)

    cdef _load_from_file(self, TString vectors_filename):
        self._storage_impl = new TDenseVectorStorage[i8](TString(vectors_filename), self._dimension)

    def __dealloc__(self):
        del self._storage_impl

    def _get_item(self, id):
        return np.array(<i8[:self._dimension]>self._storage_impl.GetItem(id))

    def _get_num_items(self):
        return self._storage_impl.GetNumItems()


cdef class _DenseI32VectorStorage:
    cdef TDenseVectorStorage[i32]* _storage_impl
    cdef size_t _dimension

    def __cinit__(self, vectors_filename = None, dimension = 0, bin_data = None, array_data = None):
        self._dimension = dimension
        if vectors_filename is not None:
            self._load_from_file(to_arcadia_string(vectors_filename))
        if bin_data is not None:
            self._load_from_bytearray(bin_data)
        if array_data is not None:
            self._load_from_array(array_data)


    cdef _load_from_array(self, np.int32_t[:, ::1] data):
        self._storage_impl = new TDenseVectorStorage[i32](TBlob.NoCopy(<const char*> &data[0,0], data.size * sizeof(np.int32_t)), self._dimension)

    cdef _load_from_bytearray(self, data):
        self._storage_impl = new TDenseVectorStorage[i32](TBlob.NoCopy(<const char*> data, len(data)), self._dimension)

    cdef _load_from_file(self, TString vectors_filename):
        self._storage_impl = new TDenseVectorStorage[i32](TString(vectors_filename), self._dimension)

    def __dealloc__(self):
        del self._storage_impl

    def _get_item(self, id):
        return np.array(<i32[:self._dimension]>self._storage_impl.GetItem(id))

    def _get_num_items(self):
        return self._storage_impl.GetNumItems()


cdef class _HnswDenseVectorIndex:
    cdef TBlob _index_blob
    cdef THnswIndexBase* _index_impl
    cdef EDistance _distance

    def __init__(self, EDistance distance):
        self._index_impl = NULL
        self._distance = distance

    def __dealloc__(self):
        if self._index_impl != NULL:
            del self._index_impl

    def _load(self, index_path):
        if self._index_impl != NULL:
            del self._index_impl
        self._index_blob = LoadIndex(to_arcadia_string(index_path))
        self._index_impl = new THnswIndexBase(self._index_blob)

    def _load_from_bytes(self, index_bin_data):
        if self._index_impl != NULL:
            del self._index_impl
        self._index_blob = TBlob.NoCopy(<const char*> index_bin_data, len(index_bin_data))
        self._index_impl = new THnswIndexBase(self._index_blob)

    def _save(self, index_path):
        SaveIndex(self._index_blob, to_arcadia_string(index_path))


cdef class _HnswDenseFloatVectorIndex(_HnswDenseVectorIndex):
    cdef _DenseFloatVectorStorage _storage

    def __init__(self, _DenseFloatVectorStorage storage, EDistance distance):
        self._storage = storage
        super(_HnswDenseFloatVectorIndex, self).__init__(distance)

    def _build(self, json_options):
        if self._index_impl != NULL:
            del self._index_impl
        cdef TString options = to_arcadia_string(json_options)
        SetPythonInterruptHandler()
        try:
            self._index_blob = BuildDenseVectorIndex[float](options, self._storage._storage_impl, self._distance)
            self._index_impl = new THnswIndexBase(self._index_blob)
        finally:
            ResetPythonInterruptHandler()

    def _get_nearest(self, query, top_size, search_neighborhood_size, distance_calc_limit):
        cdef float [:] q = np.ascontiguousarray(query, dtype=np.float32)
        return _get_nearest_neighbors_float(self._index_impl, &q[0], top_size, search_neighborhood_size,
                                            distance_calc_limit, self._storage._storage_impl, self._distance)

    def _kneighbors(self, X, size_t n_neighbors, bool_t return_distance, EDistance distance,
                    size_t search_neighborhood_size, size_t distance_calc_limit):
        cdef np.float32_t[:, ::1] queries = np.ascontiguousarray(X, dtype=np.float32)
        cdef np.ndarray neigh_ind = np.empty((queries.shape[0], n_neighbors), dtype=np.uint32)
        cdef ui32* neigh_ind_ptr = <ui32*>neigh_ind.data
        cdef np.ndarray neigh_dist
        cdef void* neigh_dist_ptr = NULL
        cdef PyObject* neigh_dist_dtype
        if return_distance:
            neigh_dist_dtype = GetDistanceResultType[np.float32_t](distance)
            py_neigh_dist_dtype = <object>neigh_dist_dtype
            Py_DECREF(py_neigh_dist_dtype)
            neigh_dist = np.empty(
                (queries.shape[0], n_neighbors),
                dtype=np.dtype(py_neigh_dist_dtype)
            )
            neigh_dist_ptr = <void*>neigh_dist.data
        else:
            neigh_dist = None

        with nogil:
            KNeighbors[np.float32_t](
                self._index_impl,
                <const float*>&queries[0,0],
                queries.shape[0],
                n_neighbors,
                search_neighborhood_size,
                distance_calc_limit,
                self._storage._storage_impl,
                distance,
                neigh_ind_ptr,
                neigh_dist_ptr)
        if return_distance:
            return neigh_dist, neigh_ind
        else:
            return neigh_ind


cdef class _HnswDenseI8VectorIndex(_HnswDenseVectorIndex):
    cdef _DenseI8VectorStorage _storage

    def __init__(self, _DenseI8VectorStorage storage, EDistance distance):
        self._storage = storage
        super(_HnswDenseI8VectorIndex, self).__init__(distance)

    def _build(self, json_options):
        if self._index_impl != NULL:
            del self._index_impl
        cdef TString options = to_arcadia_string(json_options)
        SetPythonInterruptHandler()
        try:
            self._index_blob = BuildDenseVectorIndex[i8](options, self._storage._storage_impl, self._distance)
            self._index_impl = new THnswIndexBase(self._index_blob)
        finally:
            ResetPythonInterruptHandler()

    def _get_nearest(self, query, top_size, search_neighborhood_size, distance_calc_limit):
        cdef i8 [:] q = np.ascontiguousarray(query, dtype=np.int8)
        return _get_nearest_neighbors_i8(self._index_impl, &q[0], top_size, search_neighborhood_size,
                                         distance_calc_limit, self._storage._storage_impl, self._distance)

    def _kneighbors(self, X, size_t n_neighbors, bool_t return_distance, EDistance distance,
                    size_t search_neighborhood_size, size_t distance_calc_limit):
        cdef np.int8_t[:, ::1] queries = np.ascontiguousarray(X, dtype=np.int8)
        cdef np.ndarray neigh_ind = np.empty((queries.shape[0], n_neighbors), dtype=np.uint32)
        cdef ui32* neigh_ind_ptr = <ui32*>neigh_ind.data
        cdef void* neigh_dist_ptr = NULL
        cdef np.ndarray neigh_dist
        cdef PyObject* neigh_dist_dtype
        if return_distance:
            neigh_dist_dtype = GetDistanceResultType[np.int8_t](distance)
            py_neigh_dist_dtype = <object>neigh_dist_dtype
            Py_DECREF(py_neigh_dist_dtype)
            neigh_dist = np.empty(
                (queries.shape[0], n_neighbors),
                dtype=np.dtype(py_neigh_dist_dtype)
            )
            neigh_dist_ptr = <void*>neigh_dist.data
        else:
            neigh_dist = None

        with nogil:
            KNeighbors[i8](
                self._index_impl,
                &queries[0,0],
                queries.shape[0],
                n_neighbors,
                search_neighborhood_size,
                distance_calc_limit,
                self._storage._storage_impl,
                distance,
                neigh_ind_ptr,
                neigh_dist_ptr)
        if return_distance:
            return neigh_dist, neigh_ind
        else:
            return neigh_ind


cdef class _HnswDenseI32VectorIndex(_HnswDenseVectorIndex):
    cdef _DenseI32VectorStorage _storage

    def __init__(self, _DenseI32VectorStorage storage, EDistance distance):
        self._storage = storage
        super(_HnswDenseI32VectorIndex, self).__init__(distance)

    def _build(self, json_options):
        if self._index_impl != NULL:
            del self._index_impl
        cdef TString options = to_arcadia_string(json_options)
        SetPythonInterruptHandler()
        try:
            self._index_blob = BuildDenseVectorIndex[i32](options, self._storage._storage_impl, self._distance)
            self._index_impl = new THnswIndexBase(self._index_blob)
        finally:
            ResetPythonInterruptHandler()

    def _get_nearest(self, query, top_size, search_neighborhood_size, distance_calc_limit):
        cdef i32 [:] q = np.ascontiguousarray(query, dtype=np.int32)
        return _get_nearest_neighbors_i32(self._index_impl, &q[0], top_size, search_neighborhood_size,
                                          distance_calc_limit, self._storage._storage_impl, self._distance)

    def _kneighbors(self, X, size_t n_neighbors, bool_t return_distance, EDistance distance,
                    size_t search_neighborhood_size, size_t distance_calc_limit):
        cdef np.int32_t[:, ::1] queries = np.ascontiguousarray(X, dtype=np.int32)
        cdef np.ndarray neigh_ind = np.empty((queries.shape[0], n_neighbors), dtype=np.uint32)
        cdef ui32* neigh_ind_ptr = <ui32*>neigh_ind.data
        cdef void* neigh_dist_ptr = NULL
        cdef np.ndarray neigh_dist
        cdef PyObject* neigh_dist_dtype
        if return_distance:
            neigh_dist_dtype = GetDistanceResultType[i32](distance)
            py_neigh_dist_dtype = <object>neigh_dist_dtype
            Py_DECREF(py_neigh_dist_dtype)
            neigh_dist = np.empty(
                (queries.shape[0], n_neighbors),
                dtype=np.dtype(py_neigh_dist_dtype)
            )
            neigh_dist_ptr = <void*>neigh_dist.data
        else:
            neigh_dist = None

        with nogil:
            KNeighbors[i32](
                self._index_impl,
                <i32*>&queries[0,0],
                queries.shape[0],
                n_neighbors,
                search_neighborhood_size,
                distance_calc_limit,
                self._storage._storage_impl,
                distance,
                neigh_ind_ptr,
                neigh_dist_ptr)
        if return_distance:
            return neigh_dist, neigh_ind
        else:
            return neigh_ind


cdef class _OnlineHnswDenseFloatVectorIndex:
    cdef size_t _dimension
    cdef PyOnlineHnswDenseVectorIndex[float]* _online_index

    def __init__(self, dimension, EDistance distance, json_options):
        self._dimension = dimension
        self._online_index = new PyOnlineHnswDenseVectorIndex[float](to_arcadia_string(json_options), dimension, distance)

    def __dealloc__(self):
        if self._online_index != NULL:
            del self._online_index

    def _get_nearest_neighbors(self, query, top_size):
        cdef float [:] q = np.ascontiguousarray(query, dtype=np.float32)
        py_neighbors = <object>self._online_index.GetNearestNeighbors(&q[0], top_size)
        Py_DECREF(py_neighbors)
        return py_neighbors

    def _get_nearest_neighbors_and_add_item(self, query):
        cdef float [:] q = np.ascontiguousarray(query, dtype=np.float32)
        py_neighbors = <object>self._online_index.GetNearestNeighborsAndAddItem(&q[0])
        Py_DECREF(py_neighbors)
        return py_neighbors

    def _get_item(self, id):
        return np.array(<float[:self._dimension]>self._online_index.GetItem(id))

    def _get_num_items(self):
        return self._online_index.GetNumItems()

    def _add_item(self, item):
        cdef float [:] pointer = np.ascontiguousarray(item, dtype=np.float32)
        self._online_index.AddItem(&pointer[0])


cdef class _OnlineHnswDenseI32VectorIndex:
    cdef size_t _dimension
    cdef PyOnlineHnswDenseVectorIndex[i32]* _online_index

    def __init__(self, dimension, EDistance distance, json_options):
        self._dimension = dimension
        self._online_index = new PyOnlineHnswDenseVectorIndex[i32](to_arcadia_string(json_options), dimension, distance)

    def __dealloc__(self):
        if self._online_index != NULL:
            del self._online_index

    def _get_nearest_neighbors(self, query, top_size):
        cdef i32 [:] q = np.ascontiguousarray(query, dtype=np.int32)
        py_neighbors = <object>self._online_index.GetNearestNeighbors(&q[0], top_size)
        Py_DECREF(py_neighbors)
        return py_neighbors

    def _get_nearest_neighbors_and_add_item(self, query):
        cdef i32 [:] q = np.ascontiguousarray(query, dtype=np.int32)
        py_neighbors = <object>self._online_index.GetNearestNeighborsAndAddItem(&q[0])
        Py_DECREF(py_neighbors)
        return py_neighbors

    def _get_item(self, id):
        return np.array(<i32[:self._dimension]>self._online_index.GetItem(id))

    def _get_num_items(self):
        return self._online_index.GetNumItems()

    def _add_item(self, item):
        cdef i32 [:] pointer = np.ascontiguousarray(item, dtype=np.int32)
        self._online_index.AddItem(&pointer[0])


cdef class _OnlineHnswDenseI8VectorIndex:
    cdef size_t _dimension
    cdef PyOnlineHnswDenseVectorIndex[i8]* _online_index

    def __init__(self, dimension, EDistance distance, json_options):
        self._dimension = dimension
        self._online_index = new PyOnlineHnswDenseVectorIndex[i8](to_arcadia_string(json_options), dimension, distance)

    def __dealloc__(self):
        if self._online_index != NULL:
            del self._online_index

    def _get_nearest_neighbors(self, query, top_size):
        cdef i8 [:] q = np.ascontiguousarray(query, dtype=np.int8)
        py_neighbors = <object>self._online_index.GetNearestNeighbors(&q[0], top_size)
        Py_DECREF(py_neighbors)
        return py_neighbors

    def _get_nearest_neighbors_and_add_item(self, query):
        cdef i8 [:] q = np.ascontiguousarray(query, dtype=np.int8)
        py_neighbors = <object>self._online_index.GetNearestNeighborsAndAddItem(&q[0])
        Py_DECREF(py_neighbors)
        return py_neighbors

    def _get_item(self, id):
        return np.array(<i8[:self._dimension]>self._online_index.GetItem(id))

    def _get_num_items(self):
        return self._online_index.GetNumItems()

    def _add_item(self, item):
        cdef i8 [:] pointer = np.ascontiguousarray(item, dtype=np.int8)
        self._online_index.AddItem(&pointer[0])


def _init_index(data, distance):
    """
    Returns
    -------
    (index, data) : _Dense*VectorIndex, object
    """

    data_shape = np.shape(data)
    if len(data_shape) != 2:
        raise HnswException("data is not 2-dimensional as expected")
    if (data_shape[0] == 0) or (data_shape[1] == 0):
        raise HnswException("data is empty")
    dimension = data_shape[1]

    if isinstance(data, np.ndarray):
        dtype = data.dtype
    else:
        dtype = np.dtype(type(data[0][0]))

    if dtype == np.int8:
        # numpy won't copy data if it is already in the right format
        data = np.ascontiguousarray(data, dtype=np.int8)
        storage = _DenseI8VectorStorage(dimension=dimension, array_data=data)
        index = _HnswDenseI8VectorIndex(storage, distance)
    elif dtype in [np.uint8, np.int16, np.uint16, np.int32]:
        # numpy won't copy data if it is already in the right format
        data = np.ascontiguousarray(data, dtype=np.int32)
        storage = _DenseI32VectorStorage(dimension=dimension, array_data=data)
        index = _HnswDenseI32VectorIndex(storage, distance)
    else:
        # numpy won't copy data if it is already in the right format
        data = np.ascontiguousarray(data, dtype=np.float32)
        storage = _DenseFloatVectorStorage(dimension=dimension, array_data=data)
        index = _HnswDenseFloatVectorIndex(storage, distance)

    return index, data
