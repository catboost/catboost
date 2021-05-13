LIBRARY()



SRCS(
    array_subset.cpp
    borders_io.cpp
    checksum.cpp
    compression.cpp
    connected_components.cpp
    cpu_random.cpp
    dbg_output.cpp
    dense_hash.cpp
    dense_hash_view.cpp
    distribution_helpers.cpp
    double_array_iterator.cpp
    dynamic_iterator.cpp
    element_range.cpp
    equal.cpp
    exception.cpp
    flatbuffers/guid.fbs
    guid.cpp
    hash.cpp
    int_cast.cpp
    interrupt.cpp
    map_merge.cpp
    math_utils.cpp
    matrix.cpp
    maybe.cpp
    maybe_owning_array_holder.cpp
    mem_usage.cpp
    parallel_tasks.cpp
    polymorphic_type_containers.cpp
    power_hash.cpp
    progress_helper.cpp
    permutation.cpp
    quantile.cpp
    query_info_helper.cpp
    resource_constrained_executor.cpp
    resource_holder.cpp
    restorable_rng.cpp
    sample.cpp
    serialization.cpp
    set.cpp
    short_vector_ops.cpp
    sparse_array.cpp
    vector_helpers.cpp
    wx_test.cpp
    xml_output.cpp
)

PEERDIR(
    catboost/private/libs/data_types
    catboost/private/libs/data_util
    catboost/private/libs/index_range
    catboost/libs/logging
    contrib/libs/flatbuffers
    library/cpp/binsaver
    library/cpp/containers/2d_array
    library/cpp/pop_count
    library/cpp/dbg_output
    library/cpp/digest/crc32c
    library/cpp/digest/md5
    library/cpp/json
    library/cpp/malloc/api
    library/cpp/threading/local_executor
    library/cpp/fast_exp
)

GENERATE_ENUM_SERIALIZATION(sparse_array.h)
GENERATE_ENUM_SERIALIZATION(distribution_helpers.h)

END()

RECURSE(
    parallel_sort
    parallel_sort/ut
)
