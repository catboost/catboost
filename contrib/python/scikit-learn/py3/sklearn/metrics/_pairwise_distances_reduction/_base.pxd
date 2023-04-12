cimport numpy as cnp

from cython cimport final

from ...utils._typedefs cimport ITYPE_t, DTYPE_t, SPARSE_INDEX_TYPE_t

cnp.import_array()

from ._datasets_pair cimport DatasetsPair64


cpdef DTYPE_t[::1] _sqeuclidean_row_norms64(
    X,
    ITYPE_t num_threads,
)

cdef class BaseDistancesReduction64:
    """
    Base float64 implementation template of the pairwise-distances
    reduction backends.

    Implementations inherit from this template and may override the several
    defined hooks as needed in order to easily extend functionality with
    minimal redundant code.
    """

    cdef:
        readonly DatasetsPair64 datasets_pair

        # The number of threads that can be used is stored in effective_n_threads.
        #
        # The number of threads to use in the parallelization strategy
        # (i.e. parallel_on_X or parallel_on_Y) can be smaller than effective_n_threads:
        # for small datasets, fewer threads might be needed to loop over pair of chunks.
        #
        # Hence, the number of threads that _will_ be used for looping over chunks
        # is stored in chunks_n_threads, allowing solely using what we need.
        #
        # Thus, an invariant is:
        #
        #                 chunks_n_threads <= effective_n_threads
        #
        ITYPE_t effective_n_threads
        ITYPE_t chunks_n_threads

        ITYPE_t n_samples_chunk, chunk_size

        ITYPE_t n_samples_X, X_n_samples_chunk, X_n_chunks, X_n_samples_last_chunk
        ITYPE_t n_samples_Y, Y_n_samples_chunk, Y_n_chunks, Y_n_samples_last_chunk

        bint execute_in_parallel_on_Y

    @final
    cdef void _parallel_on_X(self) nogil

    @final
    cdef void _parallel_on_Y(self) nogil

    # Placeholder methods which have to be implemented

    cdef void _compute_and_reduce_distances_on_chunks(
        self,
        ITYPE_t X_start,
        ITYPE_t X_end,
        ITYPE_t Y_start,
        ITYPE_t Y_end,
        ITYPE_t thread_num,
    ) nogil


    # Placeholder methods which can be implemented

    cdef void compute_exact_distances(self) nogil

    cdef void _parallel_on_X_parallel_init(
        self,
        ITYPE_t thread_num,
    ) nogil

    cdef void _parallel_on_X_init_chunk(
        self,
        ITYPE_t thread_num,
        ITYPE_t X_start,
        ITYPE_t X_end,
    ) nogil

    cdef void _parallel_on_X_pre_compute_and_reduce_distances_on_chunks(
        self,
        ITYPE_t X_start,
        ITYPE_t X_end,
        ITYPE_t Y_start,
        ITYPE_t Y_end,
        ITYPE_t thread_num,
    ) nogil

    cdef void _parallel_on_X_prange_iter_finalize(
        self,
        ITYPE_t thread_num,
        ITYPE_t X_start,
        ITYPE_t X_end,
    ) nogil

    cdef void _parallel_on_X_parallel_finalize(
        self,
        ITYPE_t thread_num
    ) nogil

    cdef void _parallel_on_Y_init(
        self,
    ) nogil

    cdef void _parallel_on_Y_parallel_init(
        self,
        ITYPE_t thread_num,
        ITYPE_t X_start,
        ITYPE_t X_end,
    ) nogil

    cdef void _parallel_on_Y_pre_compute_and_reduce_distances_on_chunks(
        self,
        ITYPE_t X_start,
        ITYPE_t X_end,
        ITYPE_t Y_start,
        ITYPE_t Y_end,
        ITYPE_t thread_num,
    ) nogil

    cdef void _parallel_on_Y_synchronize(
        self,
        ITYPE_t X_start,
        ITYPE_t X_end,
    ) nogil

    cdef void _parallel_on_Y_finalize(
        self,
    ) nogil

from ._datasets_pair cimport DatasetsPair32


cpdef DTYPE_t[::1] _sqeuclidean_row_norms32(
    X,
    ITYPE_t num_threads,
)

cdef class BaseDistancesReduction32:
    """
    Base float32 implementation template of the pairwise-distances
    reduction backends.

    Implementations inherit from this template and may override the several
    defined hooks as needed in order to easily extend functionality with
    minimal redundant code.
    """

    cdef:
        readonly DatasetsPair32 datasets_pair

        # The number of threads that can be used is stored in effective_n_threads.
        #
        # The number of threads to use in the parallelization strategy
        # (i.e. parallel_on_X or parallel_on_Y) can be smaller than effective_n_threads:
        # for small datasets, fewer threads might be needed to loop over pair of chunks.
        #
        # Hence, the number of threads that _will_ be used for looping over chunks
        # is stored in chunks_n_threads, allowing solely using what we need.
        #
        # Thus, an invariant is:
        #
        #                 chunks_n_threads <= effective_n_threads
        #
        ITYPE_t effective_n_threads
        ITYPE_t chunks_n_threads

        ITYPE_t n_samples_chunk, chunk_size

        ITYPE_t n_samples_X, X_n_samples_chunk, X_n_chunks, X_n_samples_last_chunk
        ITYPE_t n_samples_Y, Y_n_samples_chunk, Y_n_chunks, Y_n_samples_last_chunk

        bint execute_in_parallel_on_Y

    @final
    cdef void _parallel_on_X(self) nogil

    @final
    cdef void _parallel_on_Y(self) nogil

    # Placeholder methods which have to be implemented

    cdef void _compute_and_reduce_distances_on_chunks(
        self,
        ITYPE_t X_start,
        ITYPE_t X_end,
        ITYPE_t Y_start,
        ITYPE_t Y_end,
        ITYPE_t thread_num,
    ) nogil


    # Placeholder methods which can be implemented

    cdef void compute_exact_distances(self) nogil

    cdef void _parallel_on_X_parallel_init(
        self,
        ITYPE_t thread_num,
    ) nogil

    cdef void _parallel_on_X_init_chunk(
        self,
        ITYPE_t thread_num,
        ITYPE_t X_start,
        ITYPE_t X_end,
    ) nogil

    cdef void _parallel_on_X_pre_compute_and_reduce_distances_on_chunks(
        self,
        ITYPE_t X_start,
        ITYPE_t X_end,
        ITYPE_t Y_start,
        ITYPE_t Y_end,
        ITYPE_t thread_num,
    ) nogil

    cdef void _parallel_on_X_prange_iter_finalize(
        self,
        ITYPE_t thread_num,
        ITYPE_t X_start,
        ITYPE_t X_end,
    ) nogil

    cdef void _parallel_on_X_parallel_finalize(
        self,
        ITYPE_t thread_num
    ) nogil

    cdef void _parallel_on_Y_init(
        self,
    ) nogil

    cdef void _parallel_on_Y_parallel_init(
        self,
        ITYPE_t thread_num,
        ITYPE_t X_start,
        ITYPE_t X_end,
    ) nogil

    cdef void _parallel_on_Y_pre_compute_and_reduce_distances_on_chunks(
        self,
        ITYPE_t X_start,
        ITYPE_t X_end,
        ITYPE_t Y_start,
        ITYPE_t Y_end,
        ITYPE_t thread_num,
    ) nogil

    cdef void _parallel_on_Y_synchronize(
        self,
        ITYPE_t X_start,
        ITYPE_t X_end,
    ) nogil

    cdef void _parallel_on_Y_finalize(
        self,
    ) nogil
