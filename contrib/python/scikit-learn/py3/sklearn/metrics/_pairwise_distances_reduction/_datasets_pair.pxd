cimport numpy as cnp

from ...utils._typedefs cimport DTYPE_t, ITYPE_t, SPARSE_INDEX_TYPE_t
from ...metrics._dist_metrics cimport DistanceMetric, DistanceMetric32


cdef class DatasetsPair64:
    cdef:
        DistanceMetric distance_metric
        ITYPE_t n_features

    cdef ITYPE_t n_samples_X(self) nogil

    cdef ITYPE_t n_samples_Y(self) nogil

    cdef DTYPE_t dist(self, ITYPE_t i, ITYPE_t j) nogil

    cdef DTYPE_t surrogate_dist(self, ITYPE_t i, ITYPE_t j) nogil


cdef class DenseDenseDatasetsPair64(DatasetsPair64):
    cdef:
        const DTYPE_t[:, ::1] X
        const DTYPE_t[:, ::1] Y


cdef class SparseSparseDatasetsPair64(DatasetsPair64):
    cdef:
        const DTYPE_t[:] X_data
        const SPARSE_INDEX_TYPE_t[:] X_indices
        const SPARSE_INDEX_TYPE_t[:] X_indptr

        const DTYPE_t[:] Y_data
        const SPARSE_INDEX_TYPE_t[:] Y_indices
        const SPARSE_INDEX_TYPE_t[:] Y_indptr


cdef class SparseDenseDatasetsPair64(DatasetsPair64):
    cdef:
        const DTYPE_t[:] X_data
        const SPARSE_INDEX_TYPE_t[:] X_indices
        const SPARSE_INDEX_TYPE_t[:] X_indptr

        const DTYPE_t[:] Y_data
        const SPARSE_INDEX_TYPE_t[:] Y_indices
        ITYPE_t n_Y


cdef class DenseSparseDatasetsPair64(DatasetsPair64):
    cdef:
        # As distance metrics are commutative, we can simply rely
        # on the implementation of SparseDenseDatasetsPair and
        # swap arguments.
        DatasetsPair64 datasets_pair


cdef class DatasetsPair32:
    cdef:
        DistanceMetric32 distance_metric
        ITYPE_t n_features

    cdef ITYPE_t n_samples_X(self) nogil

    cdef ITYPE_t n_samples_Y(self) nogil

    cdef DTYPE_t dist(self, ITYPE_t i, ITYPE_t j) nogil

    cdef DTYPE_t surrogate_dist(self, ITYPE_t i, ITYPE_t j) nogil


cdef class DenseDenseDatasetsPair32(DatasetsPair32):
    cdef:
        const cnp.float32_t[:, ::1] X
        const cnp.float32_t[:, ::1] Y


cdef class SparseSparseDatasetsPair32(DatasetsPair32):
    cdef:
        const cnp.float32_t[:] X_data
        const SPARSE_INDEX_TYPE_t[:] X_indices
        const SPARSE_INDEX_TYPE_t[:] X_indptr

        const cnp.float32_t[:] Y_data
        const SPARSE_INDEX_TYPE_t[:] Y_indices
        const SPARSE_INDEX_TYPE_t[:] Y_indptr


cdef class SparseDenseDatasetsPair32(DatasetsPair32):
    cdef:
        const cnp.float32_t[:] X_data
        const SPARSE_INDEX_TYPE_t[:] X_indices
        const SPARSE_INDEX_TYPE_t[:] X_indptr

        const cnp.float32_t[:] Y_data
        const SPARSE_INDEX_TYPE_t[:] Y_indices
        ITYPE_t n_Y


cdef class DenseSparseDatasetsPair32(DatasetsPair32):
    cdef:
        # As distance metrics are commutative, we can simply rely
        # on the implementation of SparseDenseDatasetsPair and
        # swap arguments.
        DatasetsPair32 datasets_pair
