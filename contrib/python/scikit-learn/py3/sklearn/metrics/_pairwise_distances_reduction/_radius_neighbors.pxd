cimport numpy as cnp

from libcpp.memory cimport shared_ptr
from libcpp.vector cimport vector
from cython cimport final

from ...utils._typedefs cimport ITYPE_t, DTYPE_t

cnp.import_array()

######################
## std::vector to np.ndarray coercion
# As type covariance is not supported for C++ containers via Cython,
# we need to redefine fused types.
ctypedef fused vector_DITYPE_t:
    vector[ITYPE_t]
    vector[DTYPE_t]


ctypedef fused vector_vector_DITYPE_t:
    vector[vector[ITYPE_t]]
    vector[vector[DTYPE_t]]

cdef cnp.ndarray[object, ndim=1] coerce_vectors_to_nd_arrays(
    shared_ptr[vector_vector_DITYPE_t] vecs
)

#####################

from ._base cimport BaseDistancesReduction64
from ._middle_term_computer cimport MiddleTermComputer64

cdef class RadiusNeighbors64(BaseDistancesReduction64):
    """float64 implementation of the RadiusNeighbors."""

    cdef:
        DTYPE_t radius

        # DistanceMetric64 compute rank-preserving surrogate distance via rdist
        # which are proxies necessitating less computations.
        # We get the equivalent for the radius to be able to compare it against
        # vectors' rank-preserving surrogate distances.
        DTYPE_t r_radius

        # Neighbors indices and distances are returned as np.ndarrays of np.ndarrays.
        #
        # For this implementation, we want resizable buffers which we will wrap
        # into numpy arrays at the end. std::vector comes as a handy container
        # for interacting efficiently with resizable buffers.
        #
        # Though it is possible to access their buffer address with
        # std::vector::data, they can't be stolen: buffers lifetime
        # is tied to their std::vector and are deallocated when
        # std::vectors are.
        #
        # To solve this, we dynamically allocate std::vectors and then
        # encapsulate them in a StdVectorSentinel responsible for
        # freeing them when the associated np.ndarray is freed.
        #
        # Shared pointers (defined via shared_ptr) are use for safer memory management.
        # Unique pointers (defined via unique_ptr) can't be used as datastructures
        # are shared across threads for parallel_on_X; see _parallel_on_X_init_chunk.
        shared_ptr[vector[vector[ITYPE_t]]] neigh_indices
        shared_ptr[vector[vector[DTYPE_t]]] neigh_distances

        # Used as array of pointers to private datastructures used in threads.
        vector[shared_ptr[vector[vector[ITYPE_t]]]] neigh_indices_chunks
        vector[shared_ptr[vector[vector[DTYPE_t]]]] neigh_distances_chunks

        bint sort_results

    @final
    cdef void _merge_vectors(
        self,
        ITYPE_t idx,
        ITYPE_t num_threads,
    ) nogil


cdef class EuclideanRadiusNeighbors64(RadiusNeighbors64):
    """EuclideanDistance-specialisation of RadiusNeighbors64."""
    cdef:
        MiddleTermComputer64 middle_term_computer
        const DTYPE_t[::1] X_norm_squared
        const DTYPE_t[::1] Y_norm_squared

        bint use_squared_distances

from ._base cimport BaseDistancesReduction32
from ._middle_term_computer cimport MiddleTermComputer32

cdef class RadiusNeighbors32(BaseDistancesReduction32):
    """float32 implementation of the RadiusNeighbors."""

    cdef:
        DTYPE_t radius

        # DistanceMetric32 compute rank-preserving surrogate distance via rdist
        # which are proxies necessitating less computations.
        # We get the equivalent for the radius to be able to compare it against
        # vectors' rank-preserving surrogate distances.
        DTYPE_t r_radius

        # Neighbors indices and distances are returned as np.ndarrays of np.ndarrays.
        #
        # For this implementation, we want resizable buffers which we will wrap
        # into numpy arrays at the end. std::vector comes as a handy container
        # for interacting efficiently with resizable buffers.
        #
        # Though it is possible to access their buffer address with
        # std::vector::data, they can't be stolen: buffers lifetime
        # is tied to their std::vector and are deallocated when
        # std::vectors are.
        #
        # To solve this, we dynamically allocate std::vectors and then
        # encapsulate them in a StdVectorSentinel responsible for
        # freeing them when the associated np.ndarray is freed.
        #
        # Shared pointers (defined via shared_ptr) are use for safer memory management.
        # Unique pointers (defined via unique_ptr) can't be used as datastructures
        # are shared across threads for parallel_on_X; see _parallel_on_X_init_chunk.
        shared_ptr[vector[vector[ITYPE_t]]] neigh_indices
        shared_ptr[vector[vector[DTYPE_t]]] neigh_distances

        # Used as array of pointers to private datastructures used in threads.
        vector[shared_ptr[vector[vector[ITYPE_t]]]] neigh_indices_chunks
        vector[shared_ptr[vector[vector[DTYPE_t]]]] neigh_distances_chunks

        bint sort_results

    @final
    cdef void _merge_vectors(
        self,
        ITYPE_t idx,
        ITYPE_t num_threads,
    ) nogil


cdef class EuclideanRadiusNeighbors32(RadiusNeighbors32):
    """EuclideanDistance-specialisation of RadiusNeighbors32."""
    cdef:
        MiddleTermComputer32 middle_term_computer
        const DTYPE_t[::1] X_norm_squared
        const DTYPE_t[::1] Y_norm_squared

        bint use_squared_distances
