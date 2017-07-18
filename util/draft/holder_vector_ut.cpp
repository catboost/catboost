#include "holder_vector.h"

#include <library/unittest/registar.h>

SIMPLE_UNIT_TEST_SUITE(THolderVectorTest) {
    SIMPLE_UNIT_TEST(TestCreateEmpty) {
        THolderVector<int> ints;
        UNIT_ASSERT_EQUAL(ints.Size(), 0);
        UNIT_ASSERT(!ints);
    }

    SIMPLE_UNIT_TEST(TestCreateNonEmpty) {
        THolderVector<int> ints(5);
        UNIT_ASSERT_EQUAL(ints.Size(), 5);
        UNIT_ASSERT(ints);

        for (size_t i = 0; i < ints.Size(); ++i) {
            UNIT_ASSERT_EQUAL(ints[i], (int*)nullptr);
        }
    }

    SIMPLE_UNIT_TEST(TestResetValue) {
        THolderVector<int> ints;
        ints.PushBack(new int(0));
        ints.PushBack(new int(1));
        UNIT_ASSERT_VALUES_EQUAL(*ints[0], 0);
        UNIT_ASSERT_VALUES_EQUAL(*ints[1], 1);
        ints.Reset(0, new int(2));
        ints.Reset(1, new int(3));
        UNIT_ASSERT_VALUES_EQUAL(*ints[0], 2);
        UNIT_ASSERT_VALUES_EQUAL(*ints[1], 3);
    }

    SIMPLE_UNIT_TEST(TestResetNoValue) {
        THolderVector<int> ints;
        ints.Resize(1);
        UNIT_ASSERT_EQUAL(ints[0], (int*)nullptr);
        ints.Reset(0, new int(1));
        UNIT_ASSERT_UNEQUAL(ints[0], (int*)nullptr);
        UNIT_ASSERT_VALUES_EQUAL(*ints[0], 1);
    }

    SIMPLE_UNIT_TEST(TestResetSmartPtr) {
        THolderVector<int> ints;
        ints.Resize(2);

        THolder<int> holder(new int(1));
        ints.Reset(0, holder);
        UNIT_ASSERT_VALUES_EQUAL(*ints[0], 1);
        UNIT_ASSERT(!holder);

        TAutoPtr<int> autoPtr(new int(2));
        ints.Reset(1, autoPtr);
        UNIT_ASSERT_VALUES_EQUAL(*ints[1], 2);
        UNIT_ASSERT(!autoPtr);
    }

    SIMPLE_UNIT_TEST(TestSwap) {
        THolderVector<int> v1;
        v1.PushBack(new int(1));

        THolderVector<int> v2;
        v1.Swap(v2);
        UNIT_ASSERT(v1.empty() && v2.size() == 1 && *v2.front() == 1);
    }

    SIMPLE_UNIT_TEST(TestUniquePtr) {
        THolderVector<TString> v;
        std::unique_ptr<TString> str(new TString("hello"));
        v.PushBack(std::move(str));
        UNIT_ASSERT(v.Size() == 1);
        UNIT_ASSERT(str.get() == nullptr);
    }
}
