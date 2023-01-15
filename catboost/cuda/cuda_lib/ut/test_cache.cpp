#include <catboost/cuda/cuda_lib/cache.h>
#include <library/cpp/testing/unittest/registar.h>
#include <iostream>

using namespace std;

Y_UNIT_TEST_SUITE(TScopedCacheTest) {
    template <class T>
    struct TSimpleScope: public TGuidHolder {
        T& Scope;
        TSimpleScope(T& scope)
            : Scope(scope)
        {
        }
    };

    Y_UNIT_TEST(TestCache) {
        TScopedCacheHolder scopedCache;

        ui64 uniqueId = 42;

        TVector<float> scope;
        TSimpleScope<decltype(scope)> scopeHolder(scope);
        ui64 scope2 = 222;
        TSimpleScope<decltype(scope2)> scopeHolder2(scope2);

        const ui64& cachedId = scopedCache.Cache(scopeHolder, 100500, [&]() -> ui64 {
            return uniqueId++;
        });
        UNIT_ASSERT_VALUES_EQUAL(cachedId, 42);

        const ui64& cachedId2 = scopedCache.Cache(scopeHolder, 100500, [&]() -> ui64 {
            return uniqueId++;
        });
        UNIT_ASSERT_VALUES_EQUAL(cachedId2, 42);

        const ui64& cachedId3 = scopedCache.Cache(scopeHolder, 1005001, [&]() -> ui64 {
            return uniqueId * 100;
        });

        UNIT_ASSERT_VALUES_EQUAL(cachedId3, 43 * 100);

        const ui64& cachedId4 = scopedCache.Cache(scopeHolder, 100500, [&]() -> ui64 {
            return uniqueId++;
        });
        UNIT_ASSERT_VALUES_EQUAL(cachedId4, 42);
        UNIT_ASSERT_VALUES_EQUAL(&cachedId4, &cachedId);

        const ui64& cachedIdOtherScope = scopedCache.Cache(scopeHolder2, 100500, [&]() -> ui64 {
            return uniqueId++;
        });
        UNIT_ASSERT_VALUES_EQUAL(cachedIdOtherScope, 43);
        UNIT_ASSERT_VALUES_EQUAL(cachedId, 42);
        const ui64& cachedId5 = scopedCache.Cache(scopeHolder, 100500, [&]() -> ui64 {
            return uniqueId++;
        });

        UNIT_ASSERT_VALUES_EQUAL(cachedId5, 42);
    }
}
