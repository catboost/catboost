#include "ci_string.h"

#include <library/cpp/unittest/registar.h>

#include <util/generic/string_ut.h>

class TCaseStringTest: public TTestBase, private TStringTestImpl<TCiString, TTestData<char>> {
public:
    void TestSpecial() {
        TCiString ss = Data._0123456(); // type 'TCiString' is used as is
        size_t hash_val = ss.hash();
        UNIT_ASSERT(hash_val == 1489244);
    }

public:
    UNIT_TEST_SUITE(TCaseStringTest);
    UNIT_TEST(TestOperators);
    UNIT_TEST(TestOperatorsCI);

    UNIT_TEST(TestSpecial);
    UNIT_TEST_SUITE_END();
};

UNIT_TEST_SUITE_REGISTRATION(TCaseStringTest);
