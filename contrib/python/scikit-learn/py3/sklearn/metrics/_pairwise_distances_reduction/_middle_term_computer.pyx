from libcpp.vector cimport vector

from ...utils._cython_blas cimport (
  BLAS_Order,
  BLAS_Trans,
  NoTrans,
  RowMajor,
  Trans,
  _gemm,
)
from ...utils._typedefs cimport float64_t, float32_t, int32_t, intp_t

# TODO: change for `libcpp.algorithm.fill` once Cython 3 is used
# Introduction in Cython:
#
# https://github.com/cython/cython/blob/05059e2a9b89bf6738a7750b905057e5b1e3fe2e/Cython/Includes/libcpp/algorithm.pxd#L50 #noqa
cdef extern from "<algorithm>" namespace "std" nogil:
    void fill[Iter, T](Iter first, Iter last, const T& value) except + #noqa

import numpy as np
from scipy.sparse import issparse, csr_matrix


cdef void _middle_term_sparse_sparse_64(
    const float64_t[:] X_data,
    const int32_t[:] X_indices,
    const int32_t[:] X_indptr,
    intp_t X_start,
    intp_t X_end,
    const float64_t[:] Y_data,
    const int32_t[:] Y_indices,
    const int32_t[:] Y_indptr,
    intp_t Y_start,
    intp_t Y_end,
    float64_t * D,
) noexcept nogil:
    # This routine assumes that D points to the first element of a
    # zeroed buffer of length at least equal to n_X × n_Y, conceptually
    # representing a 2-d C-ordered array.
    cdef:
        intp_t i, j, k
        intp_t n_X = X_end - X_start
        intp_t n_Y = Y_end - Y_start
        intp_t x_col, x_ptr, y_col, y_ptr

    for i in range(n_X):
        for x_ptr in range(X_indptr[X_start+i], X_indptr[X_start+i+1]):
            x_col = X_indices[x_ptr]
            for j in range(n_Y):
                k = i * n_Y + j
                for y_ptr in range(Y_indptr[Y_start+j], Y_indptr[Y_start+j+1]):
                    y_col = Y_indices[y_ptr]
                    if x_col == y_col:
                        D[k] += -2 * X_data[x_ptr] * Y_data[y_ptr]


cdef void _middle_term_sparse_dense_64(
    const float64_t[:] X_data,
    const int32_t[:] X_indices,
    const int32_t[:] X_indptr,
    intp_t X_start,
    intp_t X_end,
    const float64_t[:, ::1] Y,
    intp_t Y_start,
    intp_t Y_end,
    bint c_ordered_middle_term,
    float64_t * dist_middle_terms,
) noexcept nogil:
    # This routine assumes that dist_middle_terms is a pointer to the first element
    # of a buffer filled with zeros of length at least equal to n_X × n_Y, conceptually
    # representing a 2-d C-ordered of F-ordered array.
    cdef:
        intp_t i, j, k
        intp_t n_X = X_end - X_start
        intp_t n_Y = Y_end - Y_start
        intp_t X_i_col_idx, X_i_ptr, Y_j_col_idx, Y_j_ptr

    for i in range(n_X):
        for j in range(n_Y):
            k = i * n_Y + j if c_ordered_middle_term else j * n_X + i
            for X_i_ptr in range(X_indptr[X_start+i], X_indptr[X_start+i+1]):
                X_i_col_idx = X_indices[X_i_ptr]
                dist_middle_terms[k] += -2 * X_data[X_i_ptr] * Y[Y_start + j, X_i_col_idx]


cdef class MiddleTermComputer64:
    """Helper class to compute a Euclidean distance matrix in chunks.

    This is an abstract base class that is further specialized depending
    on the type of data (dense or sparse).

    `EuclideanDistance` subclasses relies on the squared Euclidean
    distances between chunks of vectors X_c and Y_c using the
    following decomposition for the (i,j) pair :


         ||X_c_i - Y_c_j||² = ||X_c_i||² - 2 X_c_i.Y_c_j^T + ||Y_c_j||²


    This helper class is in charge of wrapping the common logic to compute
    the middle term, i.e. `- 2 X_c_i.Y_c_j^T`.
    """

    @classmethod
    def get_for(
        cls,
        X,
        Y,
        effective_n_threads,
        chunks_n_threads,
        dist_middle_terms_chunks_size,
        n_features,
        chunk_size,
    ) -> MiddleTermComputer64:
        """Return the MiddleTermComputer implementation for the given arguments.

        Parameters
        ----------
        X : ndarray or CSR sparse matrix of shape (n_samples_X, n_features)
            Input data.
            If provided as a ndarray, it must be C-contiguous.

        Y : ndarray or CSR sparse matrix of shape (n_samples_Y, n_features)
            Input data.
            If provided as a ndarray, it must be C-contiguous.

        Returns
        -------
        middle_term_computer: MiddleTermComputer64
            The suited MiddleTermComputer64 implementation.
        """
        X_is_sparse = issparse(X)
        Y_is_sparse = issparse(Y)

        if not X_is_sparse and not Y_is_sparse:
            return DenseDenseMiddleTermComputer64(
                X,
                Y,
                effective_n_threads,
                chunks_n_threads,
                dist_middle_terms_chunks_size,
                n_features,
                chunk_size,
            )
        if X_is_sparse and Y_is_sparse:
            return SparseSparseMiddleTermComputer64(
                X,
                Y,
                effective_n_threads,
                chunks_n_threads,
                dist_middle_terms_chunks_size,
                n_features,
                chunk_size,
            )
        if X_is_sparse and not Y_is_sparse:
            return SparseDenseMiddleTermComputer64(
                X,
                Y,
                effective_n_threads,
                chunks_n_threads,
                dist_middle_terms_chunks_size,
                n_features,
                chunk_size,
                c_ordered_middle_term=True
            )
        if not X_is_sparse and Y_is_sparse:
            # NOTE: The Dense-Sparse case is implement via the Sparse-Dense case.
            #
            # To do so:
            #    - X (dense) and Y (sparse) are swapped
            #    - the distance middle term is seen as F-ordered for consistency
            #      (c_ordered_middle_term = False)
            return SparseDenseMiddleTermComputer64(
                # Mind that X and Y are swapped here.
                Y,
                X,
                effective_n_threads,
                chunks_n_threads,
                dist_middle_terms_chunks_size,
                n_features,
                chunk_size,
                c_ordered_middle_term=False,
            )
        raise NotImplementedError(
            "X and Y must be CSR sparse matrices or numpy arrays."
        )

    @classmethod
    def unpack_csr_matrix(cls, X: csr_matrix):
        """Ensure that the CSR matrix is indexed with np.int32."""
        X_data = np.asarray(X.data, dtype=np.float64)
        X_indices = np.asarray(X.indices, dtype=np.int32)
        X_indptr = np.asarray(X.indptr, dtype=np.int32)
        return X_data, X_indices, X_indptr

    def __init__(
        self,
        intp_t effective_n_threads,
        intp_t chunks_n_threads,
        intp_t dist_middle_terms_chunks_size,
        intp_t n_features,
        intp_t chunk_size,
    ):
        self.effective_n_threads = effective_n_threads
        self.chunks_n_threads = chunks_n_threads
        self.dist_middle_terms_chunks_size = dist_middle_terms_chunks_size
        self.n_features = n_features
        self.chunk_size = chunk_size

        self.dist_middle_terms_chunks = vector[vector[float64_t]](self.effective_n_threads)

    cdef void _parallel_on_X_pre_compute_and_reduce_distances_on_chunks(
        self,
        intp_t X_start,
        intp_t X_end,
        intp_t Y_start,
        intp_t Y_end,
        intp_t thread_num,
    ) noexcept nogil:
        return

    cdef void _parallel_on_X_parallel_init(self, intp_t thread_num) noexcept nogil:
        self.dist_middle_terms_chunks[thread_num].resize(self.dist_middle_terms_chunks_size)

    cdef void _parallel_on_X_init_chunk(
        self,
        intp_t thread_num,
        intp_t X_start,
        intp_t X_end,
    ) noexcept nogil:
        return

    cdef void _parallel_on_Y_init(self) noexcept nogil:
        for thread_num in range(self.chunks_n_threads):
            self.dist_middle_terms_chunks[thread_num].resize(
                self.dist_middle_terms_chunks_size
            )

    cdef void _parallel_on_Y_parallel_init(
        self,
        intp_t thread_num,
        intp_t X_start,
        intp_t X_end,
    ) noexcept nogil:
        return

    cdef void _parallel_on_Y_pre_compute_and_reduce_distances_on_chunks(
        self,
        intp_t X_start,
        intp_t X_end,
        intp_t Y_start,
        intp_t Y_end,
        intp_t thread_num
    ) noexcept nogil:
        return

    cdef float64_t * _compute_dist_middle_terms(
        self,
        intp_t X_start,
        intp_t X_end,
        intp_t Y_start,
        intp_t Y_end,
        intp_t thread_num,
    ) noexcept nogil:
        return NULL


cdef class DenseDenseMiddleTermComputer64(MiddleTermComputer64):
    """Computes the middle term of the Euclidean distance between two chunked dense matrices
    X_c and Y_c.

                        dist_middle_terms = - 2 X_c_i.Y_c_j^T

    This class use the BLAS gemm routine to perform the dot product of each chunks
    of the distance matrix with improved arithmetic intensity and vector instruction (SIMD).
    """

    def __init__(
        self,
        const float64_t[:, ::1] X,
        const float64_t[:, ::1] Y,
        intp_t effective_n_threads,
        intp_t chunks_n_threads,
        intp_t dist_middle_terms_chunks_size,
        intp_t n_features,
        intp_t chunk_size,
    ):
        super().__init__(
            effective_n_threads,
            chunks_n_threads,
            dist_middle_terms_chunks_size,
            n_features,
            chunk_size,
        )
        self.X = X
        self.Y = Y

    cdef void _parallel_on_X_pre_compute_and_reduce_distances_on_chunks(
        self,
        intp_t X_start,
        intp_t X_end,
        intp_t Y_start,
        intp_t Y_end,
        intp_t thread_num,
    ) noexcept nogil:
        return

    cdef void _parallel_on_X_init_chunk(
        self,
        intp_t thread_num,
        intp_t X_start,
        intp_t X_end,
    ) noexcept nogil:
        return

    cdef void _parallel_on_Y_parallel_init(
        self,
        intp_t thread_num,
        intp_t X_start,
        intp_t X_end,
    ) noexcept nogil:
        return

    cdef void _parallel_on_Y_pre_compute_and_reduce_distances_on_chunks(
        self,
        intp_t X_start,
        intp_t X_end,
        intp_t Y_start,
        intp_t Y_end,
        intp_t thread_num
    ) noexcept nogil:
        return

    cdef float64_t * _compute_dist_middle_terms(
        self,
        intp_t X_start,
        intp_t X_end,
        intp_t Y_start,
        intp_t Y_end,
        intp_t thread_num,
    ) noexcept nogil:
        cdef:
            float64_t *dist_middle_terms = self.dist_middle_terms_chunks[thread_num].data()

            # Careful: LDA, LDB and LDC are given for F-ordered arrays
            # in BLAS documentations, for instance:
            # https://www.netlib.org/lapack/explore-html/db/dc9/group__single__blas__level3_gafe51bacb54592ff5de056acabd83c260.html #noqa
            #
            # Here, we use their counterpart values to work with C-ordered arrays.
            BLAS_Order order = RowMajor
            BLAS_Trans ta = NoTrans
            BLAS_Trans tb = Trans
            intp_t m = X_end - X_start
            intp_t n = Y_end - Y_start
            intp_t K = self.n_features
            float64_t alpha = - 2.
            # Casting for A and B to remove the const is needed because APIs exposed via
            # scipy.linalg.cython_blas aren't reflecting the arguments' const qualifier.
            # See: https://github.com/scipy/scipy/issues/14262
            float64_t * A = <float64_t *> &self.X[X_start, 0]
            float64_t * B = <float64_t *> &self.Y[Y_start, 0]
            intp_t lda = self.n_features
            intp_t ldb = self.n_features
            float64_t beta = 0.
            intp_t ldc = Y_end - Y_start

        # dist_middle_terms = `-2 * X[X_start:X_end] @ Y[Y_start:Y_end].T`
        _gemm(order, ta, tb, m, n, K, alpha, A, lda, B, ldb, beta, dist_middle_terms, ldc)

        return dist_middle_terms


cdef class SparseSparseMiddleTermComputer64(MiddleTermComputer64):
    """Middle term of the Euclidean distance between two chunked CSR matrices.

    The result is return as a contiguous array.

            dist_middle_terms = - 2 X_c_i.Y_c_j^T

    The logic of the computation is wrapped in the routine _middle_term_sparse_sparse_64.
    This routine iterates over the data, indices and indptr arrays of the sparse matrices without
    densifying them.
    """

    def __init__(
        self,
        X,
        Y,
        intp_t effective_n_threads,
        intp_t chunks_n_threads,
        intp_t dist_middle_terms_chunks_size,
        intp_t n_features,
        intp_t chunk_size,
    ):
        super().__init__(
            effective_n_threads,
            chunks_n_threads,
            dist_middle_terms_chunks_size,
            n_features,
            chunk_size,
        )
        self.X_data, self.X_indices, self.X_indptr = self.unpack_csr_matrix(X)
        self.Y_data, self.Y_indices, self.Y_indptr = self.unpack_csr_matrix(Y)

    cdef void _parallel_on_X_pre_compute_and_reduce_distances_on_chunks(
        self,
        intp_t X_start,
        intp_t X_end,
        intp_t Y_start,
        intp_t Y_end,
        intp_t thread_num,
    ) noexcept nogil:
        # Flush the thread dist_middle_terms_chunks to 0.0
        fill(
            self.dist_middle_terms_chunks[thread_num].begin(),
            self.dist_middle_terms_chunks[thread_num].end(),
            0.0,
        )

    cdef void _parallel_on_Y_pre_compute_and_reduce_distances_on_chunks(
        self,
        intp_t X_start,
        intp_t X_end,
        intp_t Y_start,
        intp_t Y_end,
        intp_t thread_num,
    ) noexcept nogil:
        # Flush the thread dist_middle_terms_chunks to 0.0
        fill(
            self.dist_middle_terms_chunks[thread_num].begin(),
            self.dist_middle_terms_chunks[thread_num].end(),
            0.0,
        )

    cdef float64_t * _compute_dist_middle_terms(
        self,
        intp_t X_start,
        intp_t X_end,
        intp_t Y_start,
        intp_t Y_end,
        intp_t thread_num,
    ) noexcept nogil:
        cdef:
            float64_t *dist_middle_terms = (
                self.dist_middle_terms_chunks[thread_num].data()
            )

        _middle_term_sparse_sparse_64(
            self.X_data,
            self.X_indices,
            self.X_indptr,
            X_start,
            X_end,
            self.Y_data,
            self.Y_indices,
            self.Y_indptr,
            Y_start,
            Y_end,
            dist_middle_terms,
        )

        return dist_middle_terms

cdef class SparseDenseMiddleTermComputer64(MiddleTermComputer64):
    """Middle term of the Euclidean distance between chunks of a CSR matrix and a np.ndarray.

    The logic of the computation is wrapped in the routine _middle_term_sparse_dense_64.
    This routine iterates over the data, indices and indptr arrays of the sparse matrices
    without densifying them.
    """

    def __init__(
        self,
        X,
        Y,
        intp_t effective_n_threads,
        intp_t chunks_n_threads,
        intp_t dist_middle_terms_chunks_size,
        intp_t n_features,
        intp_t chunk_size,
        bint c_ordered_middle_term,
    ):
        super().__init__(
            effective_n_threads,
            chunks_n_threads,
            dist_middle_terms_chunks_size,
            n_features,
            chunk_size,
        )
        self.X_data, self.X_indices, self.X_indptr = self.unpack_csr_matrix(X)
        self.Y = Y
        self.c_ordered_middle_term = c_ordered_middle_term

    cdef void _parallel_on_X_pre_compute_and_reduce_distances_on_chunks(
        self,
        intp_t X_start,
        intp_t X_end,
        intp_t Y_start,
        intp_t Y_end,
        intp_t thread_num,
    ) noexcept nogil:
        # Fill the thread's dist_middle_terms_chunks with 0.0 before
        # computing its elements in _compute_dist_middle_terms.
        fill(
            self.dist_middle_terms_chunks[thread_num].begin(),
            self.dist_middle_terms_chunks[thread_num].end(),
            0.0,
        )

    cdef void _parallel_on_Y_pre_compute_and_reduce_distances_on_chunks(
        self,
        intp_t X_start,
        intp_t X_end,
        intp_t Y_start,
        intp_t Y_end,
        intp_t thread_num,
    ) noexcept nogil:
        # Fill the thread's dist_middle_terms_chunks with 0.0 before
        # computing its elements in _compute_dist_middle_terms.
        fill(
            self.dist_middle_terms_chunks[thread_num].begin(),
            self.dist_middle_terms_chunks[thread_num].end(),
            0.0,
        )

    cdef float64_t * _compute_dist_middle_terms(
        self,
        intp_t X_start,
        intp_t X_end,
        intp_t Y_start,
        intp_t Y_end,
        intp_t thread_num,
    ) noexcept nogil:
        cdef:
            float64_t *dist_middle_terms = (
                self.dist_middle_terms_chunks[thread_num].data()
            )

        # For the dense-sparse case, we use the sparse-dense case
        # with dist_middle_terms seen as F-ordered.
        # Hence we swap indices pointers here.
        if not self.c_ordered_middle_term:
            X_start, Y_start = Y_start, X_start
            X_end, Y_end = Y_end, X_end

        _middle_term_sparse_dense_64(
            self.X_data,
            self.X_indices,
            self.X_indptr,
            X_start,
            X_end,
            self.Y,
            Y_start,
            Y_end,
            self.c_ordered_middle_term,
            dist_middle_terms,
        )

        return dist_middle_terms

cdef void _middle_term_sparse_dense_32(
    const float64_t[:] X_data,
    const int32_t[:] X_indices,
    const int32_t[:] X_indptr,
    intp_t X_start,
    intp_t X_end,
    const float32_t[:, ::1] Y,
    intp_t Y_start,
    intp_t Y_end,
    bint c_ordered_middle_term,
    float64_t * dist_middle_terms,
) noexcept nogil:
    # This routine assumes that dist_middle_terms is a pointer to the first element
    # of a buffer filled with zeros of length at least equal to n_X × n_Y, conceptually
    # representing a 2-d C-ordered of F-ordered array.
    cdef:
        intp_t i, j, k
        intp_t n_X = X_end - X_start
        intp_t n_Y = Y_end - Y_start
        intp_t X_i_col_idx, X_i_ptr, Y_j_col_idx, Y_j_ptr

    for i in range(n_X):
        for j in range(n_Y):
            k = i * n_Y + j if c_ordered_middle_term else j * n_X + i
            for X_i_ptr in range(X_indptr[X_start+i], X_indptr[X_start+i+1]):
                X_i_col_idx = X_indices[X_i_ptr]
                dist_middle_terms[k] += -2 * X_data[X_i_ptr] * Y[Y_start + j, X_i_col_idx]


cdef class MiddleTermComputer32:
    """Helper class to compute a Euclidean distance matrix in chunks.

    This is an abstract base class that is further specialized depending
    on the type of data (dense or sparse).

    `EuclideanDistance` subclasses relies on the squared Euclidean
    distances between chunks of vectors X_c and Y_c using the
    following decomposition for the (i,j) pair :


         ||X_c_i - Y_c_j||² = ||X_c_i||² - 2 X_c_i.Y_c_j^T + ||Y_c_j||²


    This helper class is in charge of wrapping the common logic to compute
    the middle term, i.e. `- 2 X_c_i.Y_c_j^T`.
    """

    @classmethod
    def get_for(
        cls,
        X,
        Y,
        effective_n_threads,
        chunks_n_threads,
        dist_middle_terms_chunks_size,
        n_features,
        chunk_size,
    ) -> MiddleTermComputer32:
        """Return the MiddleTermComputer implementation for the given arguments.

        Parameters
        ----------
        X : ndarray or CSR sparse matrix of shape (n_samples_X, n_features)
            Input data.
            If provided as a ndarray, it must be C-contiguous.

        Y : ndarray or CSR sparse matrix of shape (n_samples_Y, n_features)
            Input data.
            If provided as a ndarray, it must be C-contiguous.

        Returns
        -------
        middle_term_computer: MiddleTermComputer32
            The suited MiddleTermComputer32 implementation.
        """
        X_is_sparse = issparse(X)
        Y_is_sparse = issparse(Y)

        if not X_is_sparse and not Y_is_sparse:
            return DenseDenseMiddleTermComputer32(
                X,
                Y,
                effective_n_threads,
                chunks_n_threads,
                dist_middle_terms_chunks_size,
                n_features,
                chunk_size,
            )
        if X_is_sparse and Y_is_sparse:
            return SparseSparseMiddleTermComputer32(
                X,
                Y,
                effective_n_threads,
                chunks_n_threads,
                dist_middle_terms_chunks_size,
                n_features,
                chunk_size,
            )
        if X_is_sparse and not Y_is_sparse:
            return SparseDenseMiddleTermComputer32(
                X,
                Y,
                effective_n_threads,
                chunks_n_threads,
                dist_middle_terms_chunks_size,
                n_features,
                chunk_size,
                c_ordered_middle_term=True
            )
        if not X_is_sparse and Y_is_sparse:
            # NOTE: The Dense-Sparse case is implement via the Sparse-Dense case.
            #
            # To do so:
            #    - X (dense) and Y (sparse) are swapped
            #    - the distance middle term is seen as F-ordered for consistency
            #      (c_ordered_middle_term = False)
            return SparseDenseMiddleTermComputer32(
                # Mind that X and Y are swapped here.
                Y,
                X,
                effective_n_threads,
                chunks_n_threads,
                dist_middle_terms_chunks_size,
                n_features,
                chunk_size,
                c_ordered_middle_term=False,
            )
        raise NotImplementedError(
            "X and Y must be CSR sparse matrices or numpy arrays."
        )

    @classmethod
    def unpack_csr_matrix(cls, X: csr_matrix):
        """Ensure that the CSR matrix is indexed with np.int32."""
        X_data = np.asarray(X.data, dtype=np.float64)
        X_indices = np.asarray(X.indices, dtype=np.int32)
        X_indptr = np.asarray(X.indptr, dtype=np.int32)
        return X_data, X_indices, X_indptr

    def __init__(
        self,
        intp_t effective_n_threads,
        intp_t chunks_n_threads,
        intp_t dist_middle_terms_chunks_size,
        intp_t n_features,
        intp_t chunk_size,
    ):
        self.effective_n_threads = effective_n_threads
        self.chunks_n_threads = chunks_n_threads
        self.dist_middle_terms_chunks_size = dist_middle_terms_chunks_size
        self.n_features = n_features
        self.chunk_size = chunk_size

        self.dist_middle_terms_chunks = vector[vector[float64_t]](self.effective_n_threads)

    cdef void _parallel_on_X_pre_compute_and_reduce_distances_on_chunks(
        self,
        intp_t X_start,
        intp_t X_end,
        intp_t Y_start,
        intp_t Y_end,
        intp_t thread_num,
    ) noexcept nogil:
        return

    cdef void _parallel_on_X_parallel_init(self, intp_t thread_num) noexcept nogil:
        self.dist_middle_terms_chunks[thread_num].resize(self.dist_middle_terms_chunks_size)

    cdef void _parallel_on_X_init_chunk(
        self,
        intp_t thread_num,
        intp_t X_start,
        intp_t X_end,
    ) noexcept nogil:
        return

    cdef void _parallel_on_Y_init(self) noexcept nogil:
        for thread_num in range(self.chunks_n_threads):
            self.dist_middle_terms_chunks[thread_num].resize(
                self.dist_middle_terms_chunks_size
            )

    cdef void _parallel_on_Y_parallel_init(
        self,
        intp_t thread_num,
        intp_t X_start,
        intp_t X_end,
    ) noexcept nogil:
        return

    cdef void _parallel_on_Y_pre_compute_and_reduce_distances_on_chunks(
        self,
        intp_t X_start,
        intp_t X_end,
        intp_t Y_start,
        intp_t Y_end,
        intp_t thread_num
    ) noexcept nogil:
        return

    cdef float64_t * _compute_dist_middle_terms(
        self,
        intp_t X_start,
        intp_t X_end,
        intp_t Y_start,
        intp_t Y_end,
        intp_t thread_num,
    ) noexcept nogil:
        return NULL


cdef class DenseDenseMiddleTermComputer32(MiddleTermComputer32):
    """Computes the middle term of the Euclidean distance between two chunked dense matrices
    X_c and Y_c.

                        dist_middle_terms = - 2 X_c_i.Y_c_j^T

    This class use the BLAS gemm routine to perform the dot product of each chunks
    of the distance matrix with improved arithmetic intensity and vector instruction (SIMD).
    """

    def __init__(
        self,
        const float32_t[:, ::1] X,
        const float32_t[:, ::1] Y,
        intp_t effective_n_threads,
        intp_t chunks_n_threads,
        intp_t dist_middle_terms_chunks_size,
        intp_t n_features,
        intp_t chunk_size,
    ):
        super().__init__(
            effective_n_threads,
            chunks_n_threads,
            dist_middle_terms_chunks_size,
            n_features,
            chunk_size,
        )
        self.X = X
        self.Y = Y
        # We populate the buffer for upcasting chunks of X and Y from float32 to float64.
        self.X_c_upcast = vector[vector[float64_t]](self.effective_n_threads)
        self.Y_c_upcast = vector[vector[float64_t]](self.effective_n_threads)

        upcast_buffer_n_elements = self.chunk_size * n_features

        for thread_num in range(self.effective_n_threads):
            self.X_c_upcast[thread_num].resize(upcast_buffer_n_elements)
            self.Y_c_upcast[thread_num].resize(upcast_buffer_n_elements)

    cdef void _parallel_on_X_pre_compute_and_reduce_distances_on_chunks(
        self,
        intp_t X_start,
        intp_t X_end,
        intp_t Y_start,
        intp_t Y_end,
        intp_t thread_num,
    ) noexcept nogil:
        cdef:
            intp_t i, j
            intp_t n_chunk_samples = Y_end - Y_start

        # Upcasting Y_c=Y[Y_start:Y_end, :] from float32 to float64
        for i in range(n_chunk_samples):
            for j in range(self.n_features):
                self.Y_c_upcast[thread_num][i * self.n_features + j] = <float64_t> self.Y[Y_start + i, j]

    cdef void _parallel_on_X_init_chunk(
        self,
        intp_t thread_num,
        intp_t X_start,
        intp_t X_end,
    ) noexcept nogil:
        cdef:
            intp_t i, j
            intp_t n_chunk_samples = X_end - X_start

        # Upcasting X_c=X[X_start:X_end, :] from float32 to float64
        for i in range(n_chunk_samples):
            for j in range(self.n_features):
                self.X_c_upcast[thread_num][i * self.n_features + j] = <float64_t> self.X[X_start + i, j]

    cdef void _parallel_on_Y_parallel_init(
        self,
        intp_t thread_num,
        intp_t X_start,
        intp_t X_end,
    ) noexcept nogil:
        cdef:
            intp_t i, j
            intp_t n_chunk_samples = X_end - X_start

        # Upcasting X_c=X[X_start:X_end, :] from float32 to float64
        for i in range(n_chunk_samples):
            for j in range(self.n_features):
                self.X_c_upcast[thread_num][i * self.n_features + j] = <float64_t> self.X[X_start + i, j]

    cdef void _parallel_on_Y_pre_compute_and_reduce_distances_on_chunks(
        self,
        intp_t X_start,
        intp_t X_end,
        intp_t Y_start,
        intp_t Y_end,
        intp_t thread_num
    ) noexcept nogil:
        cdef:
            intp_t i, j
            intp_t n_chunk_samples = Y_end - Y_start

        # Upcasting Y_c=Y[Y_start:Y_end, :] from float32 to float64
        for i in range(n_chunk_samples):
            for j in range(self.n_features):
                self.Y_c_upcast[thread_num][i * self.n_features + j] = <float64_t> self.Y[Y_start + i, j]

    cdef float64_t * _compute_dist_middle_terms(
        self,
        intp_t X_start,
        intp_t X_end,
        intp_t Y_start,
        intp_t Y_end,
        intp_t thread_num,
    ) noexcept nogil:
        cdef:
            float64_t *dist_middle_terms = self.dist_middle_terms_chunks[thread_num].data()

            # Careful: LDA, LDB and LDC are given for F-ordered arrays
            # in BLAS documentations, for instance:
            # https://www.netlib.org/lapack/explore-html/db/dc9/group__single__blas__level3_gafe51bacb54592ff5de056acabd83c260.html #noqa
            #
            # Here, we use their counterpart values to work with C-ordered arrays.
            BLAS_Order order = RowMajor
            BLAS_Trans ta = NoTrans
            BLAS_Trans tb = Trans
            intp_t m = X_end - X_start
            intp_t n = Y_end - Y_start
            intp_t K = self.n_features
            float64_t alpha = - 2.
            float64_t * A = self.X_c_upcast[thread_num].data()
            float64_t * B = self.Y_c_upcast[thread_num].data()
            intp_t lda = self.n_features
            intp_t ldb = self.n_features
            float64_t beta = 0.
            intp_t ldc = Y_end - Y_start

        # dist_middle_terms = `-2 * X[X_start:X_end] @ Y[Y_start:Y_end].T`
        _gemm(order, ta, tb, m, n, K, alpha, A, lda, B, ldb, beta, dist_middle_terms, ldc)

        return dist_middle_terms


cdef class SparseSparseMiddleTermComputer32(MiddleTermComputer32):
    """Middle term of the Euclidean distance between two chunked CSR matrices.

    The result is return as a contiguous array.

            dist_middle_terms = - 2 X_c_i.Y_c_j^T

    The logic of the computation is wrapped in the routine _middle_term_sparse_sparse_64.
    This routine iterates over the data, indices and indptr arrays of the sparse matrices without
    densifying them.
    """

    def __init__(
        self,
        X,
        Y,
        intp_t effective_n_threads,
        intp_t chunks_n_threads,
        intp_t dist_middle_terms_chunks_size,
        intp_t n_features,
        intp_t chunk_size,
    ):
        super().__init__(
            effective_n_threads,
            chunks_n_threads,
            dist_middle_terms_chunks_size,
            n_features,
            chunk_size,
        )
        self.X_data, self.X_indices, self.X_indptr = self.unpack_csr_matrix(X)
        self.Y_data, self.Y_indices, self.Y_indptr = self.unpack_csr_matrix(Y)

    cdef void _parallel_on_X_pre_compute_and_reduce_distances_on_chunks(
        self,
        intp_t X_start,
        intp_t X_end,
        intp_t Y_start,
        intp_t Y_end,
        intp_t thread_num,
    ) noexcept nogil:
        # Flush the thread dist_middle_terms_chunks to 0.0
        fill(
            self.dist_middle_terms_chunks[thread_num].begin(),
            self.dist_middle_terms_chunks[thread_num].end(),
            0.0,
        )

    cdef void _parallel_on_Y_pre_compute_and_reduce_distances_on_chunks(
        self,
        intp_t X_start,
        intp_t X_end,
        intp_t Y_start,
        intp_t Y_end,
        intp_t thread_num,
    ) noexcept nogil:
        # Flush the thread dist_middle_terms_chunks to 0.0
        fill(
            self.dist_middle_terms_chunks[thread_num].begin(),
            self.dist_middle_terms_chunks[thread_num].end(),
            0.0,
        )

    cdef float64_t * _compute_dist_middle_terms(
        self,
        intp_t X_start,
        intp_t X_end,
        intp_t Y_start,
        intp_t Y_end,
        intp_t thread_num,
    ) noexcept nogil:
        cdef:
            float64_t *dist_middle_terms = (
                self.dist_middle_terms_chunks[thread_num].data()
            )

        _middle_term_sparse_sparse_64(
            self.X_data,
            self.X_indices,
            self.X_indptr,
            X_start,
            X_end,
            self.Y_data,
            self.Y_indices,
            self.Y_indptr,
            Y_start,
            Y_end,
            dist_middle_terms,
        )

        return dist_middle_terms

cdef class SparseDenseMiddleTermComputer32(MiddleTermComputer32):
    """Middle term of the Euclidean distance between chunks of a CSR matrix and a np.ndarray.

    The logic of the computation is wrapped in the routine _middle_term_sparse_dense_32.
    This routine iterates over the data, indices and indptr arrays of the sparse matrices
    without densifying them.
    """

    def __init__(
        self,
        X,
        Y,
        intp_t effective_n_threads,
        intp_t chunks_n_threads,
        intp_t dist_middle_terms_chunks_size,
        intp_t n_features,
        intp_t chunk_size,
        bint c_ordered_middle_term,
    ):
        super().__init__(
            effective_n_threads,
            chunks_n_threads,
            dist_middle_terms_chunks_size,
            n_features,
            chunk_size,
        )
        self.X_data, self.X_indices, self.X_indptr = self.unpack_csr_matrix(X)
        self.Y = Y
        self.c_ordered_middle_term = c_ordered_middle_term

    cdef void _parallel_on_X_pre_compute_and_reduce_distances_on_chunks(
        self,
        intp_t X_start,
        intp_t X_end,
        intp_t Y_start,
        intp_t Y_end,
        intp_t thread_num,
    ) noexcept nogil:
        # Fill the thread's dist_middle_terms_chunks with 0.0 before
        # computing its elements in _compute_dist_middle_terms.
        fill(
            self.dist_middle_terms_chunks[thread_num].begin(),
            self.dist_middle_terms_chunks[thread_num].end(),
            0.0,
        )

    cdef void _parallel_on_Y_pre_compute_and_reduce_distances_on_chunks(
        self,
        intp_t X_start,
        intp_t X_end,
        intp_t Y_start,
        intp_t Y_end,
        intp_t thread_num,
    ) noexcept nogil:
        # Fill the thread's dist_middle_terms_chunks with 0.0 before
        # computing its elements in _compute_dist_middle_terms.
        fill(
            self.dist_middle_terms_chunks[thread_num].begin(),
            self.dist_middle_terms_chunks[thread_num].end(),
            0.0,
        )

    cdef float64_t * _compute_dist_middle_terms(
        self,
        intp_t X_start,
        intp_t X_end,
        intp_t Y_start,
        intp_t Y_end,
        intp_t thread_num,
    ) noexcept nogil:
        cdef:
            float64_t *dist_middle_terms = (
                self.dist_middle_terms_chunks[thread_num].data()
            )

        # For the dense-sparse case, we use the sparse-dense case
        # with dist_middle_terms seen as F-ordered.
        # Hence we swap indices pointers here.
        if not self.c_ordered_middle_term:
            X_start, Y_start = Y_start, X_start
            X_end, Y_end = Y_end, X_end

        _middle_term_sparse_dense_32(
            self.X_data,
            self.X_indices,
            self.X_indptr,
            X_start,
            X_end,
            self.Y,
            Y_start,
            Y_end,
            self.c_ordered_middle_term,
            dist_middle_terms,
        )

        return dist_middle_terms
