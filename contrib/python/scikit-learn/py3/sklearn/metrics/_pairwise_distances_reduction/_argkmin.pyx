cimport numpy as cnp

from libc.stdlib cimport free, malloc
from libc.float cimport DBL_MAX
from cython cimport final
from cython.parallel cimport parallel, prange

from ...utils._heap cimport heap_push
from ...utils._sorting cimport simultaneous_sort
from ...utils._typedefs cimport ITYPE_t, DTYPE_t

import numpy as np
import warnings

from numbers import Integral
from scipy.sparse import issparse
from ...utils import check_array, check_scalar, _in_unstable_openblas_configuration
from ...utils.fixes import threadpool_limits
from ...utils._typedefs import ITYPE, DTYPE


cnp.import_array()

from ._base cimport (
    BaseDistancesReduction64,
    _sqeuclidean_row_norms64,
)

from ._datasets_pair cimport DatasetsPair64

from ._middle_term_computer cimport MiddleTermComputer64


cdef class ArgKmin64(BaseDistancesReduction64):
    """float64 implementation of the ArgKmin."""

    @classmethod
    def compute(
        cls,
        X,
        Y,
        ITYPE_t k,
        str metric="euclidean",
        chunk_size=None,
        dict metric_kwargs=None,
        str strategy=None,
        bint return_distance=False,
    ):
        """Compute the argkmin reduction.

        This classmethod is responsible for introspecting the arguments
        values to dispatch to the most appropriate implementation of
        :class:`ArgKmin64`.

        This allows decoupling the API entirely from the implementation details
        whilst maintaining RAII: all temporarily allocated datastructures necessary
        for the concrete implementation are therefore freed when this classmethod
        returns.

        No instance should directly be created outside of this class method.
        """
        if (
            metric in ("euclidean", "sqeuclidean")
            and not (issparse(X) ^ issparse(Y))  # "^" is the XOR operator
        ):
            # Specialized implementation of ArgKmin for the Euclidean distance
            # for the dense-dense and sparse-sparse cases.
            # This implementation computes the distances by chunk using
            # a decomposition of the Squared Euclidean distance.
            # This specialisation has an improved arithmetic intensity for both
            # the dense and sparse settings, allowing in most case speed-ups of
            # several orders of magnitude compared to the generic ArgKmin
            # implementation.
            # For more information see MiddleTermComputer.
            use_squared_distances = metric == "sqeuclidean"
            pda = EuclideanArgKmin64(
                X=X, Y=Y, k=k,
                use_squared_distances=use_squared_distances,
                chunk_size=chunk_size,
                strategy=strategy,
                metric_kwargs=metric_kwargs,
            )
        else:
            # Fall back on a generic implementation that handles most scipy
            # metrics by computing the distances between 2 vectors at a time.
            pda = ArgKmin64(
                datasets_pair=DatasetsPair64.get_for(X, Y, metric, metric_kwargs),
                k=k,
                chunk_size=chunk_size,
                strategy=strategy,
            )

        # Limit the number of threads in second level of nested parallelism for BLAS
        # to avoid threads over-subscription (in GEMM for instance).
        with threadpool_limits(limits=1, user_api="blas"):
            if pda.execute_in_parallel_on_Y:
                pda._parallel_on_Y()
            else:
                pda._parallel_on_X()

        return pda._finalize_results(return_distance)

    def __init__(
        self,
        DatasetsPair64 datasets_pair,
        chunk_size=None,
        strategy=None,
        ITYPE_t k=1,
    ):
        super().__init__(
            datasets_pair=datasets_pair,
            chunk_size=chunk_size,
            strategy=strategy,
        )
        self.k = check_scalar(k, "k", Integral, min_val=1)

        # Allocating pointers to datastructures but not the datastructures themselves.
        # There are as many pointers as effective threads.
        #
        # For the sake of explicitness:
        #   - when parallelizing on X, the pointers of those heaps are referencing
        #   (with proper offsets) addresses of the two main heaps (see below)
        #   - when parallelizing on Y, the pointers of those heaps are referencing
        #   small heaps which are thread-wise-allocated and whose content will be
        #   merged with the main heaps'.
        self.heaps_r_distances_chunks = <DTYPE_t **> malloc(
            sizeof(DTYPE_t *) * self.chunks_n_threads
        )
        self.heaps_indices_chunks = <ITYPE_t **> malloc(
            sizeof(ITYPE_t *) * self.chunks_n_threads
        )

        # Main heaps which will be returned as results by `ArgKmin64.compute`.
        self.argkmin_indices = np.full((self.n_samples_X, self.k), 0, dtype=ITYPE)
        self.argkmin_distances = np.full((self.n_samples_X, self.k), DBL_MAX, dtype=DTYPE)

    def __dealloc__(self):
        if self.heaps_indices_chunks is not NULL:
            free(self.heaps_indices_chunks)

        if self.heaps_r_distances_chunks is not NULL:
            free(self.heaps_r_distances_chunks)

    cdef void _compute_and_reduce_distances_on_chunks(
        self,
        ITYPE_t X_start,
        ITYPE_t X_end,
        ITYPE_t Y_start,
        ITYPE_t Y_end,
        ITYPE_t thread_num,
    ) nogil:
        cdef:
            ITYPE_t i, j
            ITYPE_t n_samples_X = X_end - X_start
            ITYPE_t n_samples_Y = Y_end - Y_start
            DTYPE_t *heaps_r_distances = self.heaps_r_distances_chunks[thread_num]
            ITYPE_t *heaps_indices = self.heaps_indices_chunks[thread_num]

        # Pushing the distances and their associated indices on a heap
        # which by construction will keep track of the argkmin.
        for i in range(n_samples_X):
            for j in range(n_samples_Y):
                heap_push(
                    values=heaps_r_distances + i * self.k,
                    indices=heaps_indices + i * self.k,
                    size=self.k,
                    val=self.datasets_pair.surrogate_dist(X_start + i, Y_start + j),
                    val_idx=Y_start + j,
                )

    cdef void _parallel_on_X_init_chunk(
        self,
        ITYPE_t thread_num,
        ITYPE_t X_start,
        ITYPE_t X_end,
    ) nogil:
        # As this strategy is embarrassingly parallel, we can set each
        # thread's heaps pointer to the proper position on the main heaps.
        self.heaps_r_distances_chunks[thread_num] = &self.argkmin_distances[X_start, 0]
        self.heaps_indices_chunks[thread_num] = &self.argkmin_indices[X_start, 0]

    @final
    cdef void _parallel_on_X_prange_iter_finalize(
        self,
        ITYPE_t thread_num,
        ITYPE_t X_start,
        ITYPE_t X_end,
    ) nogil:
        cdef:
            ITYPE_t idx

        # Sorting the main heaps portion associated to `X[X_start:X_end]`
        # in ascending order w.r.t the distances.
        for idx in range(X_end - X_start):
            simultaneous_sort(
                self.heaps_r_distances_chunks[thread_num] + idx * self.k,
                self.heaps_indices_chunks[thread_num] + idx * self.k,
                self.k
            )

    cdef void _parallel_on_Y_init(
        self,
    ) nogil:
        cdef:
            # Maximum number of scalar elements (the last chunks can be smaller)
            ITYPE_t heaps_size = self.X_n_samples_chunk * self.k
            ITYPE_t thread_num

        # The allocation is done in parallel for data locality purposes: this way
        # the heaps used in each threads are allocated in pages which are closer
        # to the CPU core used by the thread.
        # See comments about First Touch Placement Policy:
        # https://www.openmp.org/wp-content/uploads/openmp-webinar-vanderPas-20210318.pdf #noqa
        for thread_num in prange(self.chunks_n_threads, schedule='static', nogil=True,
                                 num_threads=self.chunks_n_threads):
            # As chunks of X are shared across threads, so must their
            # heaps. To solve this, each thread has its own heaps
            # which are then synchronised back in the main ones.
            self.heaps_r_distances_chunks[thread_num] = <DTYPE_t *> malloc(
                heaps_size * sizeof(DTYPE_t)
            )
            self.heaps_indices_chunks[thread_num] = <ITYPE_t *> malloc(
                heaps_size * sizeof(ITYPE_t)
            )

    cdef void _parallel_on_Y_parallel_init(
        self,
        ITYPE_t thread_num,
        ITYPE_t X_start,
        ITYPE_t X_end,
    ) nogil:
        # Initialising heaps (memset can't be used here)
        for idx in range(self.X_n_samples_chunk * self.k):
            self.heaps_r_distances_chunks[thread_num][idx] = DBL_MAX
            self.heaps_indices_chunks[thread_num][idx] = -1

    @final
    cdef void _parallel_on_Y_synchronize(
        self,
        ITYPE_t X_start,
        ITYPE_t X_end,
    ) nogil:
        cdef:
            ITYPE_t idx, jdx, thread_num
        with nogil, parallel(num_threads=self.effective_n_threads):
            # Synchronising the thread heaps with the main heaps.
            # This is done in parallel sample-wise (no need for locks).
            #
            # This might break each thread's data locality as each heap which
            # was allocated in a thread is being now being used in several threads.
            #
            # Still, this parallel pattern has shown to be efficient in practice.
            for idx in prange(X_end - X_start, schedule="static"):
                for thread_num in range(self.chunks_n_threads):
                    for jdx in range(self.k):
                        heap_push(
                            values=&self.argkmin_distances[X_start + idx, 0],
                            indices=&self.argkmin_indices[X_start + idx, 0],
                            size=self.k,
                            val=self.heaps_r_distances_chunks[thread_num][idx * self.k + jdx],
                            val_idx=self.heaps_indices_chunks[thread_num][idx * self.k + jdx],
                        )

    cdef void _parallel_on_Y_finalize(
        self,
    ) nogil:
        cdef:
            ITYPE_t idx, thread_num

        with nogil, parallel(num_threads=self.chunks_n_threads):
            # Deallocating temporary datastructures
            for thread_num in prange(self.chunks_n_threads, schedule='static'):
                free(self.heaps_r_distances_chunks[thread_num])
                free(self.heaps_indices_chunks[thread_num])

            # Sorting the main in ascending order w.r.t the distances.
            # This is done in parallel sample-wise (no need for locks).
            for idx in prange(self.n_samples_X, schedule='static'):
                simultaneous_sort(
                    &self.argkmin_distances[idx, 0],
                    &self.argkmin_indices[idx, 0],
                    self.k,
                )
        return

    cdef void compute_exact_distances(self) nogil:
        cdef:
            ITYPE_t i, j
            DTYPE_t[:, ::1] distances = self.argkmin_distances
        for i in prange(self.n_samples_X, schedule='static', nogil=True,
                        num_threads=self.effective_n_threads):
            for j in range(self.k):
                distances[i, j] = self.datasets_pair.distance_metric._rdist_to_dist(
                    # Guard against potential -0., causing nan production.
                    max(distances[i, j], 0.)
                )

    def _finalize_results(self, bint return_distance=False):
        if return_distance:
            # We need to recompute distances because we relied on
            # surrogate distances for the reduction.
            self.compute_exact_distances()

            # Values are returned identically to the way `KNeighborsMixin.kneighbors`
            # returns values. This is counter-intuitive but this allows not using
            # complex adaptations where `ArgKmin.compute` is called.
            return np.asarray(self.argkmin_distances), np.asarray(self.argkmin_indices)

        return np.asarray(self.argkmin_indices)


cdef class EuclideanArgKmin64(ArgKmin64):
    """EuclideanDistance-specialisation of ArgKmin64."""

    @classmethod
    def is_usable_for(cls, X, Y, metric) -> bool:
        return (ArgKmin64.is_usable_for(X, Y, metric) and
                not _in_unstable_openblas_configuration())

    def __init__(
        self,
        X,
        Y,
        ITYPE_t k,
        bint use_squared_distances=False,
        chunk_size=None,
        strategy=None,
        metric_kwargs=None,
    ):
        if (
            isinstance(metric_kwargs, dict) and
            (metric_kwargs.keys() - {"X_norm_squared", "Y_norm_squared"})
        ):
            warnings.warn(
                f"Some metric_kwargs have been passed ({metric_kwargs}) but aren't "
                f"usable for this case (EuclideanArgKmin64) and will be ignored.",
                UserWarning,
                stacklevel=3,
            )

        super().__init__(
            # The datasets pair here is used for exact distances computations
            datasets_pair=DatasetsPair64.get_for(X, Y, metric="euclidean"),
            chunk_size=chunk_size,
            strategy=strategy,
            k=k,
        )
        cdef:
            ITYPE_t dist_middle_terms_chunks_size = self.Y_n_samples_chunk * self.X_n_samples_chunk

        self.middle_term_computer = MiddleTermComputer64.get_for(
            X,
            Y,
            self.effective_n_threads,
            self.chunks_n_threads,
            dist_middle_terms_chunks_size,
            n_features=X.shape[1],
            chunk_size=self.chunk_size,
        )

        if metric_kwargs is not None and "Y_norm_squared" in metric_kwargs:
            self.Y_norm_squared = check_array(
                metric_kwargs.pop("Y_norm_squared"),
                ensure_2d=False,
                input_name="Y_norm_squared",
                dtype=np.float64,
            )
        else:
            self.Y_norm_squared = _sqeuclidean_row_norms64(
                Y,
                self.effective_n_threads,
            )

        if metric_kwargs is not None and "X_norm_squared" in metric_kwargs:
            self.X_norm_squared = check_array(
                metric_kwargs.pop("X_norm_squared"),
                ensure_2d=False,
                input_name="X_norm_squared",
                dtype=np.float64,
            )
        else:
            # Do not recompute norms if datasets are identical.
            self.X_norm_squared = (
                self.Y_norm_squared if X is Y else
                _sqeuclidean_row_norms64(
                    X,
                    self.effective_n_threads,
                )
            )

        self.use_squared_distances = use_squared_distances

    @final
    cdef void compute_exact_distances(self) nogil:
        if not self.use_squared_distances:
            ArgKmin64.compute_exact_distances(self)

    @final
    cdef void _parallel_on_X_parallel_init(
        self,
        ITYPE_t thread_num,
    ) nogil:
        ArgKmin64._parallel_on_X_parallel_init(self, thread_num)
        self.middle_term_computer._parallel_on_X_parallel_init(thread_num)

    @final
    cdef void _parallel_on_X_init_chunk(
        self,
        ITYPE_t thread_num,
        ITYPE_t X_start,
        ITYPE_t X_end,
    ) nogil:
        ArgKmin64._parallel_on_X_init_chunk(self, thread_num, X_start, X_end)
        self.middle_term_computer._parallel_on_X_init_chunk(thread_num, X_start, X_end)

    @final
    cdef void _parallel_on_X_pre_compute_and_reduce_distances_on_chunks(
        self,
        ITYPE_t X_start,
        ITYPE_t X_end,
        ITYPE_t Y_start,
        ITYPE_t Y_end,
        ITYPE_t thread_num,
    ) nogil:
        ArgKmin64._parallel_on_X_pre_compute_and_reduce_distances_on_chunks(
            self,
            X_start, X_end,
            Y_start, Y_end,
            thread_num,
        )
        self.middle_term_computer._parallel_on_X_pre_compute_and_reduce_distances_on_chunks(
            X_start, X_end, Y_start, Y_end, thread_num,
        )

    @final
    cdef void _parallel_on_Y_init(
        self,
    ) nogil:
        ArgKmin64._parallel_on_Y_init(self)
        self.middle_term_computer._parallel_on_Y_init()

    @final
    cdef void _parallel_on_Y_parallel_init(
        self,
        ITYPE_t thread_num,
        ITYPE_t X_start,
        ITYPE_t X_end,
    ) nogil:
        ArgKmin64._parallel_on_Y_parallel_init(self, thread_num, X_start, X_end)
        self.middle_term_computer._parallel_on_Y_parallel_init(thread_num, X_start, X_end)

    @final
    cdef void _parallel_on_Y_pre_compute_and_reduce_distances_on_chunks(
        self,
        ITYPE_t X_start,
        ITYPE_t X_end,
        ITYPE_t Y_start,
        ITYPE_t Y_end,
        ITYPE_t thread_num,
    ) nogil:
        ArgKmin64._parallel_on_Y_pre_compute_and_reduce_distances_on_chunks(
            self,
            X_start, X_end,
            Y_start, Y_end,
            thread_num,
        )
        self.middle_term_computer._parallel_on_Y_pre_compute_and_reduce_distances_on_chunks(
            X_start, X_end, Y_start, Y_end, thread_num
        )

    @final
    cdef void _compute_and_reduce_distances_on_chunks(
        self,
        ITYPE_t X_start,
        ITYPE_t X_end,
        ITYPE_t Y_start,
        ITYPE_t Y_end,
        ITYPE_t thread_num,
    ) nogil:
        cdef:
            ITYPE_t i, j
            DTYPE_t sqeuclidean_dist_i_j
            ITYPE_t n_X = X_end - X_start
            ITYPE_t n_Y = Y_end - Y_start
            DTYPE_t * dist_middle_terms = self.middle_term_computer._compute_dist_middle_terms(
                X_start, X_end, Y_start, Y_end, thread_num
            )
            DTYPE_t * heaps_r_distances = self.heaps_r_distances_chunks[thread_num]
            ITYPE_t * heaps_indices = self.heaps_indices_chunks[thread_num]

        # Pushing the distance and their associated indices on heaps
        # which keep tracks of the argkmin.
        for i in range(n_X):
            for j in range(n_Y):
                sqeuclidean_dist_i_j = (
                    self.X_norm_squared[i + X_start] +
                    dist_middle_terms[i * n_Y + j] +
                    self.Y_norm_squared[j + Y_start]
                )

                # Catastrophic cancellation might cause -0. to be present,
                # e.g. when computing d(x_i, y_i) when X is Y.
                sqeuclidean_dist_i_j = max(0., sqeuclidean_dist_i_j)

                heap_push(
                    values=heaps_r_distances + i * self.k,
                    indices=heaps_indices + i * self.k,
                    size=self.k,
                    val=sqeuclidean_dist_i_j,
                    val_idx=j + Y_start,
                )

from ._base cimport (
    BaseDistancesReduction32,
    _sqeuclidean_row_norms32,
)

from ._datasets_pair cimport DatasetsPair32

from ._middle_term_computer cimport MiddleTermComputer32


cdef class ArgKmin32(BaseDistancesReduction32):
    """float32 implementation of the ArgKmin."""

    @classmethod
    def compute(
        cls,
        X,
        Y,
        ITYPE_t k,
        str metric="euclidean",
        chunk_size=None,
        dict metric_kwargs=None,
        str strategy=None,
        bint return_distance=False,
    ):
        """Compute the argkmin reduction.

        This classmethod is responsible for introspecting the arguments
        values to dispatch to the most appropriate implementation of
        :class:`ArgKmin32`.

        This allows decoupling the API entirely from the implementation details
        whilst maintaining RAII: all temporarily allocated datastructures necessary
        for the concrete implementation are therefore freed when this classmethod
        returns.

        No instance should directly be created outside of this class method.
        """
        if (
            metric in ("euclidean", "sqeuclidean")
            and not (issparse(X) ^ issparse(Y))  # "^" is the XOR operator
        ):
            # Specialized implementation of ArgKmin for the Euclidean distance
            # for the dense-dense and sparse-sparse cases.
            # This implementation computes the distances by chunk using
            # a decomposition of the Squared Euclidean distance.
            # This specialisation has an improved arithmetic intensity for both
            # the dense and sparse settings, allowing in most case speed-ups of
            # several orders of magnitude compared to the generic ArgKmin
            # implementation.
            # For more information see MiddleTermComputer.
            use_squared_distances = metric == "sqeuclidean"
            pda = EuclideanArgKmin32(
                X=X, Y=Y, k=k,
                use_squared_distances=use_squared_distances,
                chunk_size=chunk_size,
                strategy=strategy,
                metric_kwargs=metric_kwargs,
            )
        else:
            # Fall back on a generic implementation that handles most scipy
            # metrics by computing the distances between 2 vectors at a time.
            pda = ArgKmin32(
                datasets_pair=DatasetsPair32.get_for(X, Y, metric, metric_kwargs),
                k=k,
                chunk_size=chunk_size,
                strategy=strategy,
            )

        # Limit the number of threads in second level of nested parallelism for BLAS
        # to avoid threads over-subscription (in GEMM for instance).
        with threadpool_limits(limits=1, user_api="blas"):
            if pda.execute_in_parallel_on_Y:
                pda._parallel_on_Y()
            else:
                pda._parallel_on_X()

        return pda._finalize_results(return_distance)

    def __init__(
        self,
        DatasetsPair32 datasets_pair,
        chunk_size=None,
        strategy=None,
        ITYPE_t k=1,
    ):
        super().__init__(
            datasets_pair=datasets_pair,
            chunk_size=chunk_size,
            strategy=strategy,
        )
        self.k = check_scalar(k, "k", Integral, min_val=1)

        # Allocating pointers to datastructures but not the datastructures themselves.
        # There are as many pointers as effective threads.
        #
        # For the sake of explicitness:
        #   - when parallelizing on X, the pointers of those heaps are referencing
        #   (with proper offsets) addresses of the two main heaps (see below)
        #   - when parallelizing on Y, the pointers of those heaps are referencing
        #   small heaps which are thread-wise-allocated and whose content will be
        #   merged with the main heaps'.
        self.heaps_r_distances_chunks = <DTYPE_t **> malloc(
            sizeof(DTYPE_t *) * self.chunks_n_threads
        )
        self.heaps_indices_chunks = <ITYPE_t **> malloc(
            sizeof(ITYPE_t *) * self.chunks_n_threads
        )

        # Main heaps which will be returned as results by `ArgKmin32.compute`.
        self.argkmin_indices = np.full((self.n_samples_X, self.k), 0, dtype=ITYPE)
        self.argkmin_distances = np.full((self.n_samples_X, self.k), DBL_MAX, dtype=DTYPE)

    def __dealloc__(self):
        if self.heaps_indices_chunks is not NULL:
            free(self.heaps_indices_chunks)

        if self.heaps_r_distances_chunks is not NULL:
            free(self.heaps_r_distances_chunks)

    cdef void _compute_and_reduce_distances_on_chunks(
        self,
        ITYPE_t X_start,
        ITYPE_t X_end,
        ITYPE_t Y_start,
        ITYPE_t Y_end,
        ITYPE_t thread_num,
    ) nogil:
        cdef:
            ITYPE_t i, j
            ITYPE_t n_samples_X = X_end - X_start
            ITYPE_t n_samples_Y = Y_end - Y_start
            DTYPE_t *heaps_r_distances = self.heaps_r_distances_chunks[thread_num]
            ITYPE_t *heaps_indices = self.heaps_indices_chunks[thread_num]

        # Pushing the distances and their associated indices on a heap
        # which by construction will keep track of the argkmin.
        for i in range(n_samples_X):
            for j in range(n_samples_Y):
                heap_push(
                    values=heaps_r_distances + i * self.k,
                    indices=heaps_indices + i * self.k,
                    size=self.k,
                    val=self.datasets_pair.surrogate_dist(X_start + i, Y_start + j),
                    val_idx=Y_start + j,
                )

    cdef void _parallel_on_X_init_chunk(
        self,
        ITYPE_t thread_num,
        ITYPE_t X_start,
        ITYPE_t X_end,
    ) nogil:
        # As this strategy is embarrassingly parallel, we can set each
        # thread's heaps pointer to the proper position on the main heaps.
        self.heaps_r_distances_chunks[thread_num] = &self.argkmin_distances[X_start, 0]
        self.heaps_indices_chunks[thread_num] = &self.argkmin_indices[X_start, 0]

    @final
    cdef void _parallel_on_X_prange_iter_finalize(
        self,
        ITYPE_t thread_num,
        ITYPE_t X_start,
        ITYPE_t X_end,
    ) nogil:
        cdef:
            ITYPE_t idx

        # Sorting the main heaps portion associated to `X[X_start:X_end]`
        # in ascending order w.r.t the distances.
        for idx in range(X_end - X_start):
            simultaneous_sort(
                self.heaps_r_distances_chunks[thread_num] + idx * self.k,
                self.heaps_indices_chunks[thread_num] + idx * self.k,
                self.k
            )

    cdef void _parallel_on_Y_init(
        self,
    ) nogil:
        cdef:
            # Maximum number of scalar elements (the last chunks can be smaller)
            ITYPE_t heaps_size = self.X_n_samples_chunk * self.k
            ITYPE_t thread_num

        # The allocation is done in parallel for data locality purposes: this way
        # the heaps used in each threads are allocated in pages which are closer
        # to the CPU core used by the thread.
        # See comments about First Touch Placement Policy:
        # https://www.openmp.org/wp-content/uploads/openmp-webinar-vanderPas-20210318.pdf #noqa
        for thread_num in prange(self.chunks_n_threads, schedule='static', nogil=True,
                                 num_threads=self.chunks_n_threads):
            # As chunks of X are shared across threads, so must their
            # heaps. To solve this, each thread has its own heaps
            # which are then synchronised back in the main ones.
            self.heaps_r_distances_chunks[thread_num] = <DTYPE_t *> malloc(
                heaps_size * sizeof(DTYPE_t)
            )
            self.heaps_indices_chunks[thread_num] = <ITYPE_t *> malloc(
                heaps_size * sizeof(ITYPE_t)
            )

    cdef void _parallel_on_Y_parallel_init(
        self,
        ITYPE_t thread_num,
        ITYPE_t X_start,
        ITYPE_t X_end,
    ) nogil:
        # Initialising heaps (memset can't be used here)
        for idx in range(self.X_n_samples_chunk * self.k):
            self.heaps_r_distances_chunks[thread_num][idx] = DBL_MAX
            self.heaps_indices_chunks[thread_num][idx] = -1

    @final
    cdef void _parallel_on_Y_synchronize(
        self,
        ITYPE_t X_start,
        ITYPE_t X_end,
    ) nogil:
        cdef:
            ITYPE_t idx, jdx, thread_num
        with nogil, parallel(num_threads=self.effective_n_threads):
            # Synchronising the thread heaps with the main heaps.
            # This is done in parallel sample-wise (no need for locks).
            #
            # This might break each thread's data locality as each heap which
            # was allocated in a thread is being now being used in several threads.
            #
            # Still, this parallel pattern has shown to be efficient in practice.
            for idx in prange(X_end - X_start, schedule="static"):
                for thread_num in range(self.chunks_n_threads):
                    for jdx in range(self.k):
                        heap_push(
                            values=&self.argkmin_distances[X_start + idx, 0],
                            indices=&self.argkmin_indices[X_start + idx, 0],
                            size=self.k,
                            val=self.heaps_r_distances_chunks[thread_num][idx * self.k + jdx],
                            val_idx=self.heaps_indices_chunks[thread_num][idx * self.k + jdx],
                        )

    cdef void _parallel_on_Y_finalize(
        self,
    ) nogil:
        cdef:
            ITYPE_t idx, thread_num

        with nogil, parallel(num_threads=self.chunks_n_threads):
            # Deallocating temporary datastructures
            for thread_num in prange(self.chunks_n_threads, schedule='static'):
                free(self.heaps_r_distances_chunks[thread_num])
                free(self.heaps_indices_chunks[thread_num])

            # Sorting the main in ascending order w.r.t the distances.
            # This is done in parallel sample-wise (no need for locks).
            for idx in prange(self.n_samples_X, schedule='static'):
                simultaneous_sort(
                    &self.argkmin_distances[idx, 0],
                    &self.argkmin_indices[idx, 0],
                    self.k,
                )
        return

    cdef void compute_exact_distances(self) nogil:
        cdef:
            ITYPE_t i, j
            DTYPE_t[:, ::1] distances = self.argkmin_distances
        for i in prange(self.n_samples_X, schedule='static', nogil=True,
                        num_threads=self.effective_n_threads):
            for j in range(self.k):
                distances[i, j] = self.datasets_pair.distance_metric._rdist_to_dist(
                    # Guard against potential -0., causing nan production.
                    max(distances[i, j], 0.)
                )

    def _finalize_results(self, bint return_distance=False):
        if return_distance:
            # We need to recompute distances because we relied on
            # surrogate distances for the reduction.
            self.compute_exact_distances()

            # Values are returned identically to the way `KNeighborsMixin.kneighbors`
            # returns values. This is counter-intuitive but this allows not using
            # complex adaptations where `ArgKmin.compute` is called.
            return np.asarray(self.argkmin_distances), np.asarray(self.argkmin_indices)

        return np.asarray(self.argkmin_indices)


cdef class EuclideanArgKmin32(ArgKmin32):
    """EuclideanDistance-specialisation of ArgKmin32."""

    @classmethod
    def is_usable_for(cls, X, Y, metric) -> bool:
        return (ArgKmin32.is_usable_for(X, Y, metric) and
                not _in_unstable_openblas_configuration())

    def __init__(
        self,
        X,
        Y,
        ITYPE_t k,
        bint use_squared_distances=False,
        chunk_size=None,
        strategy=None,
        metric_kwargs=None,
    ):
        if (
            isinstance(metric_kwargs, dict) and
            (metric_kwargs.keys() - {"X_norm_squared", "Y_norm_squared"})
        ):
            warnings.warn(
                f"Some metric_kwargs have been passed ({metric_kwargs}) but aren't "
                f"usable for this case (EuclideanArgKmin64) and will be ignored.",
                UserWarning,
                stacklevel=3,
            )

        super().__init__(
            # The datasets pair here is used for exact distances computations
            datasets_pair=DatasetsPair32.get_for(X, Y, metric="euclidean"),
            chunk_size=chunk_size,
            strategy=strategy,
            k=k,
        )
        cdef:
            ITYPE_t dist_middle_terms_chunks_size = self.Y_n_samples_chunk * self.X_n_samples_chunk

        self.middle_term_computer = MiddleTermComputer32.get_for(
            X,
            Y,
            self.effective_n_threads,
            self.chunks_n_threads,
            dist_middle_terms_chunks_size,
            n_features=X.shape[1],
            chunk_size=self.chunk_size,
        )

        if metric_kwargs is not None and "Y_norm_squared" in metric_kwargs:
            self.Y_norm_squared = check_array(
                metric_kwargs.pop("Y_norm_squared"),
                ensure_2d=False,
                input_name="Y_norm_squared",
                dtype=np.float64,
            )
        else:
            self.Y_norm_squared = _sqeuclidean_row_norms32(
                Y,
                self.effective_n_threads,
            )

        if metric_kwargs is not None and "X_norm_squared" in metric_kwargs:
            self.X_norm_squared = check_array(
                metric_kwargs.pop("X_norm_squared"),
                ensure_2d=False,
                input_name="X_norm_squared",
                dtype=np.float64,
            )
        else:
            # Do not recompute norms if datasets are identical.
            self.X_norm_squared = (
                self.Y_norm_squared if X is Y else
                _sqeuclidean_row_norms32(
                    X,
                    self.effective_n_threads,
                )
            )

        self.use_squared_distances = use_squared_distances

    @final
    cdef void compute_exact_distances(self) nogil:
        if not self.use_squared_distances:
            ArgKmin32.compute_exact_distances(self)

    @final
    cdef void _parallel_on_X_parallel_init(
        self,
        ITYPE_t thread_num,
    ) nogil:
        ArgKmin32._parallel_on_X_parallel_init(self, thread_num)
        self.middle_term_computer._parallel_on_X_parallel_init(thread_num)

    @final
    cdef void _parallel_on_X_init_chunk(
        self,
        ITYPE_t thread_num,
        ITYPE_t X_start,
        ITYPE_t X_end,
    ) nogil:
        ArgKmin32._parallel_on_X_init_chunk(self, thread_num, X_start, X_end)
        self.middle_term_computer._parallel_on_X_init_chunk(thread_num, X_start, X_end)

    @final
    cdef void _parallel_on_X_pre_compute_and_reduce_distances_on_chunks(
        self,
        ITYPE_t X_start,
        ITYPE_t X_end,
        ITYPE_t Y_start,
        ITYPE_t Y_end,
        ITYPE_t thread_num,
    ) nogil:
        ArgKmin32._parallel_on_X_pre_compute_and_reduce_distances_on_chunks(
            self,
            X_start, X_end,
            Y_start, Y_end,
            thread_num,
        )
        self.middle_term_computer._parallel_on_X_pre_compute_and_reduce_distances_on_chunks(
            X_start, X_end, Y_start, Y_end, thread_num,
        )

    @final
    cdef void _parallel_on_Y_init(
        self,
    ) nogil:
        ArgKmin32._parallel_on_Y_init(self)
        self.middle_term_computer._parallel_on_Y_init()

    @final
    cdef void _parallel_on_Y_parallel_init(
        self,
        ITYPE_t thread_num,
        ITYPE_t X_start,
        ITYPE_t X_end,
    ) nogil:
        ArgKmin32._parallel_on_Y_parallel_init(self, thread_num, X_start, X_end)
        self.middle_term_computer._parallel_on_Y_parallel_init(thread_num, X_start, X_end)

    @final
    cdef void _parallel_on_Y_pre_compute_and_reduce_distances_on_chunks(
        self,
        ITYPE_t X_start,
        ITYPE_t X_end,
        ITYPE_t Y_start,
        ITYPE_t Y_end,
        ITYPE_t thread_num,
    ) nogil:
        ArgKmin32._parallel_on_Y_pre_compute_and_reduce_distances_on_chunks(
            self,
            X_start, X_end,
            Y_start, Y_end,
            thread_num,
        )
        self.middle_term_computer._parallel_on_Y_pre_compute_and_reduce_distances_on_chunks(
            X_start, X_end, Y_start, Y_end, thread_num
        )

    @final
    cdef void _compute_and_reduce_distances_on_chunks(
        self,
        ITYPE_t X_start,
        ITYPE_t X_end,
        ITYPE_t Y_start,
        ITYPE_t Y_end,
        ITYPE_t thread_num,
    ) nogil:
        cdef:
            ITYPE_t i, j
            DTYPE_t sqeuclidean_dist_i_j
            ITYPE_t n_X = X_end - X_start
            ITYPE_t n_Y = Y_end - Y_start
            DTYPE_t * dist_middle_terms = self.middle_term_computer._compute_dist_middle_terms(
                X_start, X_end, Y_start, Y_end, thread_num
            )
            DTYPE_t * heaps_r_distances = self.heaps_r_distances_chunks[thread_num]
            ITYPE_t * heaps_indices = self.heaps_indices_chunks[thread_num]

        # Pushing the distance and their associated indices on heaps
        # which keep tracks of the argkmin.
        for i in range(n_X):
            for j in range(n_Y):
                sqeuclidean_dist_i_j = (
                    self.X_norm_squared[i + X_start] +
                    dist_middle_terms[i * n_Y + j] +
                    self.Y_norm_squared[j + Y_start]
                )

                # Catastrophic cancellation might cause -0. to be present,
                # e.g. when computing d(x_i, y_i) when X is Y.
                sqeuclidean_dist_i_j = max(0., sqeuclidean_dist_i_j)

                heap_push(
                    values=heaps_r_distances + i * self.k,
                    indices=heaps_indices + i * self.k,
                    size=self.k,
                    val=sqeuclidean_dist_i_j,
                    val_idx=j + Y_start,
                )
