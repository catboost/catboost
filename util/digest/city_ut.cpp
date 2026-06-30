#include "city.h"

#include <library/cpp/testing/unittest/registar.h>

Y_UNIT_TEST_SUITE(TCityTest) {
    Y_UNIT_TEST(TestTemplatesCompiling) {
        TStringBuf s;
        CityHash64(s);
        CityHash64WithSeed(s, 1);
        CityHash64WithSeeds(s, 1, 2);
        CityHash128(s);
        CityHash128WithSeed(s, uint128(1, 2));
        UNIT_ASSERT(s.empty());
    }
} // Y_UNIT_TEST_SUITE(TCityTest)
