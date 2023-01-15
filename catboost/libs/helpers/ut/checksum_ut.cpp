#include <catboost/libs/helpers/checksum.h>

#include <util/generic/map.h>
#include <util/stream/output.h>
#include <util/system/types.h>

#include <array>
#include <utility>

#include <library/cpp/testing/unittest/registar.h>


using namespace NCB;


Y_UNIT_TEST_SUITE(CheckSum) {
    Y_UNIT_TEST(Test) {
        UNIT_ASSERT_VALUES_EQUAL(UpdateCheckSum(0, 0), 1214729159);
        UNIT_ASSERT_VALUES_EQUAL(UpdateCheckSum(0, std::array<int, 2>{10, 12}), 3507297803);
        UNIT_ASSERT_VALUES_EQUAL(UpdateCheckSum(0, TMap<ui32, ui32>{{12, 10}, {0, 5}}), 2630080714);
        UNIT_ASSERT_VALUES_EQUAL(
            UpdateCheckSum(0, TMap<ui32, TMap<ui32, ui32>>{{0, {{12, 10}}}, {2, {{0, 5}}}}),
            1251117058
        );
        UNIT_ASSERT_VALUES_EQUAL(UpdateCheckSum(0, TVector<int>{10, 12}), 3507297803);
        UNIT_ASSERT_VALUES_EQUAL(
            UpdateCheckSum(0, TVector<TMap<ui32, ui32>>{{{0, 1}, {2, 3}}, {{2, 11}, {3, 8}}}),
            288846871
        );
    }
}
