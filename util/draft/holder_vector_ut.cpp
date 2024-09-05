#include "holder_vector.h"

#include <library/cpp/testing/unittest/registar.h>

Y_UNIT_TEST_SUITE(THolderVectorTest) {
    Y_UNIT_TEST(TestCreateEmpty) {
        THolderVector<int> ints;
        UNIT_ASSERT_EQUAL(ints.Size(), 0);
        UNIT_ASSERT(!ints);
    }

    Y_UNIT_TEST(TestCreateNonEmpty) {
        THolderVector<int> ints(5);
        UNIT_ASSERT_EQUAL(ints.Size(), 5);
        UNIT_ASSERT(ints);

        for (size_t i = 0; i < ints.Size(); ++i) {
            UNIT_ASSERT_EQUAL(ints[i], (int*)nullptr);
        }
    }

    Y_UNIT_TEST(TestResetValue) {
        THolderVector<int> ints;
        ints.PushBack(new int(0));
        ints.PushBack(new int(1));
        UNIT_ASSERT_VALUES_EQUAL(*ints[0], 0);
        UNIT_ASSERT_VALUES_EQUAL(*ints[1], 1);
        ints.Reset(0, MakeHolder<int>(2));
        ints.Reset(1, MakeHolder<int>(3));
        UNIT_ASSERT_VALUES_EQUAL(*ints[0], 2);
        UNIT_ASSERT_VALUES_EQUAL(*ints[1], 3);
    }

    Y_UNIT_TEST(TestResetNoValue) {
        THolderVector<int> ints;
        ints.Resize(1);
        UNIT_ASSERT_EQUAL(ints[0], (int*)nullptr);
        ints.Reset(0, MakeHolder<int>(1));
        UNIT_ASSERT_UNEQUAL(ints[0], (int*)nullptr);
        UNIT_ASSERT_VALUES_EQUAL(*ints[0], 1);
    }

    Y_UNIT_TEST(TestResetSmartPtr) {
        THolderVector<int> ints;
        ints.Resize(2);

        THolder<int> holder(new int(1));
        ints.Reset(0, std::move(holder));
        UNIT_ASSERT_VALUES_EQUAL(*ints[0], 1);
        UNIT_ASSERT(!holder);
    }

    Y_UNIT_TEST(TestSwap) {
        THolderVector<int> v1;
        v1.PushBack(new int(1));

        THolderVector<int> v2;
        v1.Swap(v2);
        UNIT_ASSERT(v1.empty() && v2.size() == 1 && *v2.front() == 1);
    }

    Y_UNIT_TEST(TestUniquePtr) {
        THolderVector<TString> v;
        std::unique_ptr<TString> str(new TString("hello"));
        v.PushBack(std::move(str));
        UNIT_ASSERT(v.Size() == 1);
        UNIT_ASSERT(str.get() == nullptr);
    }
} // Y_UNIT_TEST_SUITE(THolderVectorTest)
