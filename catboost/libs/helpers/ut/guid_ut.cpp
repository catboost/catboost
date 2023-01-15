#include <catboost/libs/helpers/guid.h>

#include <library/cpp/testing/unittest/registar.h>

using namespace NCB;

static bool Equal(TArrayRef<ui32> left, TArrayRef<ui32> right) {
    return Equal(left.begin(), left.end(), right.begin());
}

Y_UNIT_TEST_SUITE(TGuidTest) {
    Y_UNIT_TEST(TestBasicProperties) {
        const char* const illegalGuid = "___ILLEGAL_GUID";
        TGuid guid;
        UNIT_ASSERT_STRINGS_EQUAL(guid.Value.data(), illegalGuid);

        guid = CreateGuid();
        TGuid guid2 = CreateGuid();

        UNIT_ASSERT_STRINGS_UNEQUAL(guid.Value.data(), illegalGuid);
        UNIT_ASSERT_STRINGS_UNEQUAL(guid2.Value.data(), illegalGuid);
        UNIT_ASSERT_STRINGS_UNEQUAL(guid.Value.data(), guid2.Value.data());

        UNIT_ASSERT(!Equal(guid.dw, guid2.dw));
    }

    Y_UNIT_TEST(TestCopyAndSwap) {
        TGuid guid1 = CreateGuid();
        TGuid guid1Base = guid1;
        UNIT_ASSERT_EQUAL(guid1, guid1Base);
        UNIT_ASSERT(guid1 == guid1Base);
        UNIT_ASSERT(Equal(guid1.dw, guid1Base.dw));

        TGuid guid2 = CreateGuid();
        TGuid guid2Base = guid2;
        UNIT_ASSERT_EQUAL(guid2, guid2Base);
        UNIT_ASSERT(guid2 == guid2Base);
        UNIT_ASSERT(Equal(guid2.dw, guid2Base.dw));

        UNIT_ASSERT(guid1 != guid2);
        UNIT_ASSERT(!Equal(guid1.dw, guid2.dw));

        guid1.Swap(guid2);
        UNIT_ASSERT(guid1 == guid2Base);
        UNIT_ASSERT(guid2 == guid1Base);
    }
}
