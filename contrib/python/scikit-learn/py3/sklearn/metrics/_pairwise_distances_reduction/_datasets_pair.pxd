from ...utils._typedefs cimport float64_t, float32_t, int32_t, intp_t
from ...metrics._dist_metrics cimport DistanceMetric64, DistanceMetric32, DistanceMetric


cdef class DatasetsPair64:
    cdef:
        DistanceMetric64 distance_metric
        intp_t n_features

    cdef intp_t n_samples_X(self) noexcept nogil

    cdef intp_t n_samples_Y(self) noexcept nogil

    cdef float64_t dist(self, intp_t i, intp_t j) noexcept nogil

    cdef float64_t surrogate_dist(self, intp_t i, intp_t j) noexcept nogil


cdef class DenseDenseDatasetsPair64(DatasetsPair64):
    cdef:
        const float64_t[:, ::1] X
        const float64_t[:, ::1] Y


cdef class SparseSparseDatasetsPair64(DatasetsPair64):
    cdef:
        const float64_t[:] X_data
        const int32_t[::1] X_indices
        const int32_t[::1] X_indptr

        const float64_t[:] Y_data
        const int32_t[::1] Y_indices
        const int32_t[::1] Y_indptr


cdef class SparseDenseDatasetsPair64(DatasetsPair64):
    cdef:
        const float64_t[:] X_data
        const int32_t[::1] X_indices
        const int32_t[::1] X_indptr

        const float64_t[:] Y_data
        const int32_t[::1] Y_indices
        intp_t n_Y


cdef class DenseSparseDatasetsPair64(DatasetsPair64):
    cdef:
        # As distance metrics are commutative, we can simply rely
        # on the implementation of SparseDenseDatasetsPair and
        # swap arguments.
        DatasetsPair64 datasets_pair


cdef class DatasetsPair32:
    cdef:
        DistanceMetric32 distance_metric
        intp_t n_features

    cdef intp_t n_samples_X(self) noexcept nogil

    cdef intp_t n_samples_Y(self) noexcept nogil

    cdef float64_t dist(self, intp_t i, intp_t j) noexcept nogil

    cdef float64_t surrogate_dist(self, intp_t i, intp_t j) noexcept nogil


cdef class DenseDenseDatasetsPair32(DatasetsPair32):
    cdef:
        const float32_t[:, ::1] X
        const float32_t[:, ::1] Y


cdef class SparseSparseDatasetsPair32(DatasetsPair32):
    cdef:
        const float32_t[:] X_data
        const int32_t[::1] X_indices
        const int32_t[::1] X_indptr

        const float32_t[:] Y_data
        const int32_t[::1] Y_indices
        const int32_t[::1] Y_indptr


cdef class SparseDenseDatasetsPair32(DatasetsPair32):
    cdef:
        const float32_t[:] X_data
        const int32_t[::1] X_indices
        const int32_t[::1] X_indptr

        const float32_t[:] Y_data
        const int32_t[::1] Y_indices
        intp_t n_Y


cdef class DenseSparseDatasetsPair32(DatasetsPair32):
    cdef:
        # As distance metrics are commutative, we can simply rely
        # on the implementation of SparseDenseDatasetsPair and
        # swap arguments.
        DatasetsPair32 datasets_pair
