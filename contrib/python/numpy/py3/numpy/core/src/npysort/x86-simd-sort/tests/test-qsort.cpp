#include "test-partial-qsort.hpp"
#include "test-qselect.hpp"
#include "test-qsort-fp.hpp"
#include "test-qsort.hpp"

using QSortTestTypes = testing::Types<uint16_t,
                                      int16_t,
                                      float,
                                      double,
                                      uint32_t,
                                      int32_t,
                                      uint64_t,
                                      int64_t>;

using QSortTestFPTypes = testing::Types<float, double>;

INSTANTIATE_TYPED_TEST_SUITE_P(T, avx512_sort, QSortTestTypes);
INSTANTIATE_TYPED_TEST_SUITE_P(T, avx512_sort_fp, QSortTestFPTypes);
INSTANTIATE_TYPED_TEST_SUITE_P(T, avx512_select, QSortTestTypes);
INSTANTIATE_TYPED_TEST_SUITE_P(T, avx512_partial_sort, QSortTestTypes);
