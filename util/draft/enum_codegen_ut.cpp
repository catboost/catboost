#include <library/unittest/registar.h>

#include "enum_codegen.h"

#include <util/string/builder.h>

#define COLOR_MAP(XX) \
    XX(RED)           \
    XX(GREEN)         \
    XX(BLUE)

enum EColor {
    COLOR_MAP(ENUM_VALUE_GEN_NO_VALUE)
};

ENUM_TO_STRING(EColor, COLOR_MAP)

#define MULTIPLIER_MAP(XX) \
    XX(GB, 9)              \
    XX(MB, 6)              \
    XX(KB, 3)

enum EMultiplier {
    MULTIPLIER_MAP(ENUM_VALUE_GEN)
};

ENUM_TO_STRING(EMultiplier, MULTIPLIER_MAP)

SIMPLE_UNIT_TEST_SUITE(EnumCodegen) {
    SIMPLE_UNIT_TEST(GenWithValue) {
        UNIT_ASSERT_VALUES_EQUAL(6, MB);
    }

    SIMPLE_UNIT_TEST(ToCString) {
        UNIT_ASSERT_VALUES_EQUAL("RED", ToCString(RED));
        UNIT_ASSERT_VALUES_EQUAL("BLUE", ToCString(BLUE));
        UNIT_ASSERT_VALUES_EQUAL("GREEN", (TStringBuilder() << GREEN));
        UNIT_ASSERT_VALUES_EQUAL("GB", ToCString(GB));
    }
}
