#include "region_packer.h"
#include <library/cpp/testing/unittest/registar.h>

template <typename TValue>
void TestPacker() {
    TValue values[] = {1, 2, 3, 42};
    TString buffer;

    TRegionPacker<TValue> p;

    using TValues = TArrayRef<TValue>;
    TValues valueRegion = TValues(values, Y_ARRAY_SIZE(values));
    size_t sz = p.MeasureLeaf(valueRegion);
    UNIT_ASSERT_VALUES_EQUAL(sz, 1 + sizeof(values));

    buffer.resize(sz);
    p.PackLeaf(buffer.begin(), valueRegion, sz);
    UNIT_ASSERT_VALUES_EQUAL(buffer[0], 4);

    p.UnpackLeaf(buffer.data(), valueRegion);
    UNIT_ASSERT_EQUAL(valueRegion.data(), (const TValue*)(buffer.begin() + 1));
    UNIT_ASSERT_EQUAL(valueRegion.size(), Y_ARRAY_SIZE(values));
    UNIT_ASSERT_EQUAL(0, memcmp(values, valueRegion.data(), sizeof(values)));
}

Y_UNIT_TEST_SUITE(RegionPacker) {
    Y_UNIT_TEST(Test0) {
        TestPacker<char>();
        TestPacker<signed char>();
        TestPacker<unsigned char>();
        TestPacker<i8>();
        TestPacker<ui8>();
        TestPacker<i16>();
        TestPacker<ui16>();
        TestPacker<i32>();
        TestPacker<ui32>();
        TestPacker<i64>();
        TestPacker<ui64>();
    }
}
