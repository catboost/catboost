#include "dense_hash.h"

#include <library/unittest/registar.h>

Y_UNIT_TEST_SUITE(TDenseHashTest) {
    Y_UNIT_TEST(TestDenseHash) {
        const ui32 elementsCount = 32;
        const ui32 sumKeysTarget = elementsCount * (elementsCount - 1) / 2;

        const ui32 addition = 20;
        const ui32 sumValuesTarget = sumKeysTarget + addition * elementsCount;

        TDenseHash<ui32, ui32> denseHash((ui32)-1);

        for (ui32 i = 0; i < elementsCount; ++i) {
            denseHash.GetMutable(i) = i + addition;
        }

        for (ui32 i = 0; i < elementsCount; ++i) {
            UNIT_ASSERT_EQUAL(i + addition, denseHash.Get(i));
            UNIT_ASSERT(denseHash.Has(i));
            UNIT_ASSERT_UNEQUAL(denseHash.FindPtr(i), nullptr);
        }

        UNIT_ASSERT(!denseHash.Has(elementsCount + 1));
        UNIT_ASSERT(!denseHash.Has(elementsCount + 10));

        UNIT_ASSERT_EQUAL(elementsCount, denseHash.Size());

        ui32 sumKeys = 0;
        ui32 sumValues = 0;
        for (auto& v : denseHash) {
            UNIT_ASSERT_EQUAL(v.first + addition, v.second);

            sumKeys += v.first;
            sumValues += v.second;
        }
        UNIT_ASSERT_EQUAL(sumKeys, sumKeysTarget);
        UNIT_ASSERT_EQUAL(sumValues, sumValuesTarget);
    }

    Y_UNIT_TEST(TestDenseHashSet) {
        const ui32 elementsCount = 32;
        const ui32 sumKeysTarget = elementsCount * (elementsCount - 1) / 2;

        TDenseHashSet<ui32> denseHashSet((ui32)-1);

        for (ui32 i = 0; i < elementsCount; ++i) {
            denseHashSet.Insert(i);
        }

        for (ui32 i = 0; i < elementsCount; ++i) {
            UNIT_ASSERT(denseHashSet.Has(i));
        }

        UNIT_ASSERT(!denseHashSet.Has(elementsCount + 1));
        UNIT_ASSERT(!denseHashSet.Has(elementsCount + 10));

        UNIT_ASSERT_EQUAL(elementsCount, denseHashSet.Size());

        ui32 sumKeys = 0;
        for (const ui32 key : denseHashSet) {
            sumKeys += key;
        }
        UNIT_ASSERT_EQUAL(sumKeys, sumKeysTarget);
    }
}
