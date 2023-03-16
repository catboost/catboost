#include "ci_string.h"

#include <util/generic/hash.h>
#include <util/generic/string_ut.h>

class TCaseStringTest: public TTestBase, private TStringTestImpl<TCiString, TTestData<char>> {
public:
    void TestSpecial() {
        TCiString ss = Data._0123456(); // type 'TCiString' is used as is
        size_t hash_val = ComputeHash(ss);
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
