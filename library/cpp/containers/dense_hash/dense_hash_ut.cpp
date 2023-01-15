#include "dense_hash.h"

#include <library/cpp/unittest/registar.h>

Y_UNIT_TEST_SUITE(TDenseHashTest) {
    Y_UNIT_TEST(TestDenseHash) {
        const ui32 elementsCount = 32;
        const ui32 sumKeysTarget = elementsCount * (elementsCount - 1) / 2;

        const ui32 addition = 20;
        const ui32 sumValuesTarget = sumKeysTarget + addition * elementsCount;

        TDenseHash<ui32, ui32> denseHash((ui32)-1);

        for (ui32 i = 0; i < elementsCount; ++i) {
            denseHash[i] = i + addition;
        }

        for (ui32 i = 0; i < elementsCount; ++i) {
            UNIT_ASSERT_EQUAL(i + addition, denseHash.Value(i, 0));
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

        auto denseHash2 = denseHash;
        UNIT_ASSERT_EQUAL(denseHash.Size(), denseHash2.Size());
        UNIT_ASSERT_EQUAL(denseHash.Capacity(), denseHash2.Capacity());
        UNIT_ASSERT_EQUAL(denseHash, denseHash2);

        denseHash2.Clear();

        denseHash2 = denseHash;
        UNIT_ASSERT_EQUAL(denseHash.Size(), denseHash2.Size());
        UNIT_ASSERT_EQUAL(denseHash.Capacity(), denseHash2.Capacity());
        UNIT_ASSERT_EQUAL(denseHash, denseHash2);

        auto denseHash3 = std::move(denseHash2);
        UNIT_ASSERT_EQUAL(denseHash.Size(), denseHash3.Size());
        UNIT_ASSERT_EQUAL(denseHash.Capacity(), denseHash3.Capacity());
        UNIT_ASSERT_EQUAL(denseHash, denseHash3);

        denseHash2 = std::move(denseHash3);
        UNIT_ASSERT_EQUAL(denseHash.Size(), denseHash2.Size());
        UNIT_ASSERT_EQUAL(denseHash.Capacity(), denseHash2.Capacity());
        UNIT_ASSERT_EQUAL(denseHash, denseHash2);
    }

    Y_UNIT_TEST(TestInsert) {
        TDenseHash<ui32, ui32> dh;
        UNIT_ASSERT(dh.Empty());

        auto p = dh.insert({ 5, 6 });
        UNIT_ASSERT(p.second);
        UNIT_ASSERT_UNEQUAL(p.first, dh.end());
        UNIT_ASSERT(!dh.Empty());
        UNIT_ASSERT_EQUAL(dh.Size(), 1);

        auto p2 = dh.insert({ 5, 26 });
        UNIT_ASSERT(!p2.second);
        UNIT_ASSERT_EQUAL(p.first, p2.first);
        UNIT_ASSERT(!dh.Empty());
        UNIT_ASSERT_EQUAL(dh.Size(), 1);
    }

    Y_UNIT_TEST(TestAtOp) {
        TDenseHash<i32, i32> dh;
        UNIT_ASSERT(dh.Empty());

        auto* p = &(dh[5] = 6);
        UNIT_ASSERT_EQUAL(*p, 6);
        UNIT_ASSERT(!dh.Empty());
        UNIT_ASSERT_EQUAL(dh.Size(), 1);

        auto* p2 = &(dh[5] = 8);
        UNIT_ASSERT_EQUAL(p, p2);
        UNIT_ASSERT_EQUAL(*p2, 8);
        UNIT_ASSERT(!dh.Empty());
        UNIT_ASSERT_EQUAL(dh.Size(), 1);
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
