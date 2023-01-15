#include <library/cpp/containers/flat_hash/flat_hash.h>

#include <library/cpp/containers/dense_hash/dense_hash.h>
#include <library/cpp/testing/benchmark/bench.h>

#include <util/random/random.h>
#include <util/generic/xrange.h>
#include <util/generic/hash.h>

namespace {

template <class Map, size_t elemCount, class... Args>
void RunLookupPositiveScalarKeysBench(::NBench::NCpu::TParams& iface, Args&&... args) {
    using key_type = i32;
    static_assert(std::is_same_v<typename Map::key_type, key_type>);
    Map hm(std::forward<Args>(args)...);

    TVector<i32> keys(elemCount);
    for (auto& k : keys) {
        k = RandomNumber<ui32>(std::numeric_limits<i32>::max());
        hm.emplace(k, 0);
    }

    for (const auto i : xrange(iface.Iterations())) {
        Y_UNUSED(i);
        for (const auto& k : keys) {
            Y_DO_NOT_OPTIMIZE_AWAY(hm[k]);
        }
    }
}

constexpr size_t TEST1_ELEM_COUNT = 10;
constexpr size_t TEST2_ELEM_COUNT = 1000;
constexpr size_t TEST3_ELEM_COUNT = 1000000;

}

/* *********************************** TEST1 ***********************************
 * Insert TEST1_ELEM_COUNT positive integers and than make lookup.
 * No init size provided for tables.
 * key_type - i32
 */

Y_CPU_BENCHMARK(Test1_fh_TFlatHashMap_LinearProbing, iface) {
    RunLookupPositiveScalarKeysBench<NFH::TFlatHashMap<i32, int>, TEST1_ELEM_COUNT>(iface);
}

/*
Y_CPU_BENCHMARK(Test1_fh_TFlatHashMap_QuadraticProbing, iface) {
    RunLookupPositiveScalarKeysBench<NFH::TFlatHashMap<i32, int, THash<i32>,
                                   std::equal_to<i32>, NFlatHash::TQuadraticProbing>, TEST1_ELEM_COUNT>(iface);
}
*/

Y_CPU_BENCHMARK(Test1_fh_TFlatHashMap_DenseProbing, iface) {
    RunLookupPositiveScalarKeysBench<NFH::TFlatHashMap<i32, int, THash<i32>,
                                   std::equal_to<i32>, NFlatHash::TDenseProbing>, TEST1_ELEM_COUNT>(iface);
}


Y_CPU_BENCHMARK(Test1_fh_TDenseHashMapStaticMarker_LinearProbing, iface) {
    RunLookupPositiveScalarKeysBench<NFH::TDenseHashMapStaticMarker<i32, int, -1, THash<i32>,
                                   std::equal_to<i32>, NFlatHash::TLinearProbing>, TEST1_ELEM_COUNT>(iface);
}

/*
Y_CPU_BENCHMARK(Test1_fh_TDenseHashMapStaticMarker_QuadraticProbing, iface) {
    RunLookupPositiveScalarKeysBench<NFH::TDenseHashMapStaticMarker<i32, int, -1, THash<i32>,
                                   std::equal_to<i32>, NFlatHash::TQuadraticProbing>, TEST1_ELEM_COUNT>(iface);
}
*/

Y_CPU_BENCHMARK(Test1_fh_TDenseHashMapStaticMarker_DenseProbing, iface) {
    RunLookupPositiveScalarKeysBench<NFH::TDenseHashMapStaticMarker<i32, int, -1>, TEST1_ELEM_COUNT>(iface);
}


Y_CPU_BENCHMARK(Test1_foreign_TDenseHash, iface) {
    RunLookupPositiveScalarKeysBench<TDenseHash<i32, int>, TEST1_ELEM_COUNT>(iface, (i32)-1);
}

Y_CPU_BENCHMARK(Test1_foreign_THashMap, iface) {
    RunLookupPositiveScalarKeysBench<THashMap<i32, int>, TEST1_ELEM_COUNT>(iface);
}

/* *********************************** TEST2 ***********************************
 * Insert TEST2_ELEM_COUNT positive integers and than make lookup.
 * No init size provided for tables.
 * key_type - i32
 */

Y_CPU_BENCHMARK(Test2_fh_TFlatHashMap_LinearProbing, iface) {
    RunLookupPositiveScalarKeysBench<NFH::TFlatHashMap<i32, int>, TEST2_ELEM_COUNT>(iface);
}

/*
Y_CPU_BENCHMARK(Test2_fh_TFlatHashMap_QuadraticProbing, iface) {
    RunLookupPositiveScalarKeysBench<NFH::TFlatHashMap<i32, int, THash<i32>,
                                   std::equal_to<i32>, NFlatHash::TQuadraticProbing>, TEST2_ELEM_COUNT>(iface);
}
*/

Y_CPU_BENCHMARK(Test2_fh_TFlatHashMap_DenseProbing, iface) {
    RunLookupPositiveScalarKeysBench<NFH::TFlatHashMap<i32, int, THash<i32>,
                                   std::equal_to<i32>, NFlatHash::TDenseProbing>, TEST2_ELEM_COUNT>(iface);
}


Y_CPU_BENCHMARK(Test2_fh_TDenseHashMapStaticMarker_LinearProbing, iface) {
    RunLookupPositiveScalarKeysBench<NFH::TDenseHashMapStaticMarker<i32, int, -1, THash<i32>,
                                   std::equal_to<i32>, NFlatHash::TLinearProbing>, TEST2_ELEM_COUNT>(iface);
}

/*
Y_CPU_BENCHMARK(Test2_fh_TDenseHashMapStaticMarker_QuadraticProbing, iface) {
    RunLookupPositiveScalarKeysBench<NFH::TDenseHashMapStaticMarker<i32, int, -1, THash<i32>,
                                   std::equal_to<i32>, NFlatHash::TQuadraticProbing>, TEST2_ELEM_COUNT>(iface);
}
*/

Y_CPU_BENCHMARK(Test2_fh_TDenseHashMapStaticMarker_DenseProbing, iface) {
    RunLookupPositiveScalarKeysBench<NFH::TDenseHashMapStaticMarker<i32, int, -1>, TEST2_ELEM_COUNT>(iface);
}


Y_CPU_BENCHMARK(Test2_foreign_TDenseHash, iface) {
    RunLookupPositiveScalarKeysBench<TDenseHash<i32, int>, TEST2_ELEM_COUNT>(iface, (i32)-1);
}

Y_CPU_BENCHMARK(Test2_foreign_THashMap, iface) {
    RunLookupPositiveScalarKeysBench<THashMap<i32, int>, TEST2_ELEM_COUNT>(iface);
}

/* *********************************** TEST3 ***********************************
 * Insert TEST2_ELEM_COUNT positive integers and than make lookup.
 * No init size provided for tables.
 * key_type - i32
 */

Y_CPU_BENCHMARK(Test3_fh_TFlatHashMap_LinearProbing, iface) {
    RunLookupPositiveScalarKeysBench<NFH::TFlatHashMap<i32, int>, TEST3_ELEM_COUNT>(iface);
}

/*
Y_CPU_BENCHMARK(Test3_fh_TFlatHashMap_QuadraticProbing, iface) {
    RunLookupPositiveScalarKeysBench<NFH::TFlatHashMap<i32, int, THash<i32>,
                                   std::equal_to<i32>, NFlatHash::TQuadraticProbing>, TEST3_ELEM_COUNT>(iface);
}
*/

Y_CPU_BENCHMARK(Test3_fh_TFlatHashMap_DenseProbing, iface) {
    RunLookupPositiveScalarKeysBench<NFH::TFlatHashMap<i32, int, THash<i32>,
                                   std::equal_to<i32>, NFlatHash::TDenseProbing>, TEST3_ELEM_COUNT>(iface);
}


Y_CPU_BENCHMARK(Test3_fh_TDenseHashMapStaticMarker_LinearProbing, iface) {
    RunLookupPositiveScalarKeysBench<NFH::TDenseHashMapStaticMarker<i32, int, -1, THash<i32>,
                                   std::equal_to<i32>, NFlatHash::TLinearProbing>, TEST3_ELEM_COUNT>(iface);
}

/*
Y_CPU_BENCHMARK(Test3_fh_TDenseHashMapStaticMarker_QuadraticProbing, iface) {
    RunLookupPositiveScalarKeysBench<NFH::TDenseHashMapStaticMarker<i32, int, -1, THash<i32>,
                                   std::equal_to<i32>, NFlatHash::TQuadraticProbing>, TEST3_ELEM_COUNT>(iface);
}
*/

Y_CPU_BENCHMARK(Test3_fh_TDenseHashMapStaticMarker_DenseProbing, iface) {
    RunLookupPositiveScalarKeysBench<NFH::TDenseHashMapStaticMarker<i32, int, -1>, TEST3_ELEM_COUNT>(iface);
}


Y_CPU_BENCHMARK(Test3_foreign_TDenseHash, iface) {
    RunLookupPositiveScalarKeysBench<TDenseHash<i32, int>, TEST3_ELEM_COUNT>(iface, (i32)-1);
}

Y_CPU_BENCHMARK(Test3_foreign_THashMap, iface) {
    RunLookupPositiveScalarKeysBench<THashMap<i32, int>, TEST3_ELEM_COUNT>(iface);
}
