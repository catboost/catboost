UNITTEST_FOR(util)



FORK_TESTS()

SRCS(
    generic/adaptor_ut.cpp
    generic/algorithm_ut.cpp
    generic/array_ref_ut.cpp
    generic/array_size_ut.cpp
    generic/bitmap_ut.cpp
    generic/bitops_ut.cpp
    generic/buffer_ut.cpp
    generic/cast_ut.cpp
    generic/chartraits_ut.cpp
    generic/deque_ut.cpp
    generic/explicit_type_ut.cpp
    generic/flags_ut.cpp
    generic/function_ut.cpp
    generic/guid_ut.cpp
    generic/hash_primes_ut.cpp
    generic/hash_ut.cpp
    generic/intrlist_ut.cpp
    generic/is_in_ut.cpp
    generic/iterator_ut.cpp
    generic/iterator_range_ut.cpp
    generic/lazy_value_ut.cpp
    generic/list_ut.cpp
    generic/map_ut.cpp
    generic/mapfindptr_ut.cpp
    generic/maybe_ut.cpp
    generic/mem_copy_ut.cpp
    generic/objects_counter_ut.cpp
    generic/ptr_ut.cpp
    generic/queue_ut.cpp
    generic/serialized_enum_ut.cpp
    generic/set_ut.cpp
    generic/singleton_ut.cpp
    generic/size_literals_ut.cpp
    generic/stack_ut.cpp
    generic/store_policy_ut.cpp
    generic/strbuf_ut.cpp
    generic/string_ut.cpp
    generic/typelist_ut.cpp
    generic/typetraits_ut.cpp
    generic/utility_ut.cpp
    generic/va_args_ut.cpp
    generic/variant_ut.cpp
    generic/vector_ut.cpp
    generic/xrange_ut.cpp
    generic/yexception_ut.c
    generic/yexception_ut.cpp
    generic/ylimits_ut.cpp
    generic/ymath_ut.cpp
    generic/scope_ut.cpp
)

INCLUDE(${ARCADIA_ROOT}/util/tests/ya_util_tests.inc)

IF (NOT OS_IOS AND NOT ARCH_PPC64LE)
    # Abseil fails to build (with linkage error) on ios and with compilation error on PowerPC
    # (somewhere in unscaledcycleclock.cc).
    PEERDIR(
        library/cpp/containers/absl_flat_hash
    )

    SRCS(
        generic/string_transparent_hash_ut.cpp
    )
ENDIF()

END()
