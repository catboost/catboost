#include "dense_hash.h"

#include <library/unittest/registar.h>

SIMPLE_UNIT_TEST_SUITE(TDenseHashTest) {
    SIMPLE_UNIT_TEST(TestDenseHash) {
        const ui32 elementsCount = 32;
        const ui32 sumKeysTarget = elementsCount * (elementsCount - 1) / 2;

        const ui32 addition = 20;
        const ui32 sumValuesTarget = sumKeysTarget + addition * elementsCount;

        TDenseHash<ui32, ui32> denseHash((ui32) -1);

        for (ui32 i = 0; i < elementsCount; ++i) {
            denseHash.GetMutable(i) = i + addition;
        }

        for (ui32 i = 0; i < elementsCount; ++i) {
            UNIT_ASSERT_VALUES_EQUAL(i + addition, denseHash.Get(i));
            UNIT_ASSERT_VALUES_EQUAL(true, denseHash.Has(i));
            UNIT_ASSERT_VALUES_EQUAL(true, denseHash.FindPtr(i) != nullptr);
        }

        UNIT_ASSERT_VALUES_EQUAL(false, denseHash.Has(elementsCount + 1));
        UNIT_ASSERT_VALUES_EQUAL(false, denseHash.Has(elementsCount + 10));

        UNIT_ASSERT_VALUES_EQUAL(elementsCount, denseHash.Size());

        ui32 sumKeys = 0;
        ui32 sumValues = 0;
        for (TDenseHash<ui32, ui32>::TIterator it : denseHash) {
            UNIT_ASSERT_VALUES_EQUAL(it.Key() + addition, it.Value());

            sumKeys += it.Key();
            sumValues += it.Value();
        }
        UNIT_ASSERT_VALUES_EQUAL(sumKeys, sumKeysTarget);
        UNIT_ASSERT_VALUES_EQUAL(sumValues, sumValuesTarget);
    }

    SIMPLE_UNIT_TEST(TestDenseHashWithConstMarker) {
        const ui32 elementsCount = 32;
        const ui32 sumKeysTarget = elementsCount * (elementsCount - 1) / 2;

        const ui32 addition = 20;
        const ui32 sumValuesTarget = sumKeysTarget + addition * elementsCount;

        TDenseHash<ui32, ui32, THash<ui32>, 50, 8, TConstIntEmptyMarker<(ui32)-1>> denseHash;

        for (ui32 i = 0; i < elementsCount; ++i) {
            denseHash.GetMutable(i) = i + addition;
        }

        for (ui32 i = 0; i < elementsCount; ++i) {
            UNIT_ASSERT_VALUES_EQUAL(i + addition, denseHash.Get(i));
            UNIT_ASSERT_VALUES_EQUAL(true, denseHash.Has(i));
            UNIT_ASSERT_VALUES_EQUAL(true, denseHash.FindPtr(i) != nullptr);
        }

        UNIT_ASSERT_VALUES_EQUAL(false, denseHash.Has(elementsCount + 1));
        UNIT_ASSERT_VALUES_EQUAL(false, denseHash.Has(elementsCount + 10));

        UNIT_ASSERT_VALUES_EQUAL(elementsCount, denseHash.Size());

        ui32 sumKeys = 0;
        ui32 sumValues = 0;
        for (auto it : denseHash) {
            UNIT_ASSERT_VALUES_EQUAL(it.Key() + addition, it.Value());

            sumKeys += it.Key();
            sumValues += it.Value();
        }
        UNIT_ASSERT_VALUES_EQUAL(sumKeys, sumKeysTarget);
        UNIT_ASSERT_VALUES_EQUAL(sumValues, sumValuesTarget);
    }

    SIMPLE_UNIT_TEST(TestDenseHashSet) {
        const ui32 elementsCount = 32;
        const ui32 sumKeysTarget = elementsCount * (elementsCount - 1) / 2;

        TDenseHashSet<ui32> denseHashSet((ui32) -1);

        for (ui32 i = 0; i < elementsCount; ++i) {
            denseHashSet.Insert(i);
        }

        for (ui32 i = 0; i < elementsCount; ++i) {
            UNIT_ASSERT_VALUES_EQUAL(true, denseHashSet.Has(i));
        }

        UNIT_ASSERT_VALUES_EQUAL(false, denseHashSet.Has(elementsCount + 1));
        UNIT_ASSERT_VALUES_EQUAL(false, denseHashSet.Has(elementsCount + 10));

        UNIT_ASSERT_VALUES_EQUAL(elementsCount, denseHashSet.Size());

        ui32 sumKeys = 0;
        for (const ui32 key : denseHashSet) {
            sumKeys += key;
        }
        UNIT_ASSERT_VALUES_EQUAL(sumKeys, sumKeysTarget);
    }
}
