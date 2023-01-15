#include <library/cpp/unittest/registar.h>
#include <library/cpp/json/flex_buffers/cvt.h>

using namespace NJson;

static auto JSON = R"({
    "a": {
        "b": [1, 2, 3],
        "c": ["x", "y", 3, "z"]
    }
})";

static auto RES = R"({ a: { b: [ 1, 2, 3 ], c: [ "x", "y", 3, "z" ] } })";

Y_UNIT_TEST_SUITE(JsonToFlex) {
    Y_UNIT_TEST(Test1) {
        auto buf = ConvertJsonToFlexBuffers(JSON);

        UNIT_ASSERT_VALUES_EQUAL(FlexToString(buf), RES);
    }
}
