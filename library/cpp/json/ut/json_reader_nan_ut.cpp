#include <library/cpp/json/json_reader.h>
#include <library/cpp/testing/unittest/registar.h>

using namespace NJson;

namespace {

constexpr TStringBuf JSON_NAN_TEST = "{ \"Value1\": 0.0, \"Value2\": 1, \"Value3\": NaN }";

}

Y_UNIT_TEST_SUITE(TJsonReaderNanTest) {
    Y_UNIT_TEST(WithoutNanTest) {
        TJsonReaderConfig cfg;
        TJsonValue out;
        // This read will fail
        UNIT_ASSERT(!ReadJsonTree(JSON_NAN_TEST, &cfg, &out, /* throwOnError */ false));

    }
    Y_UNIT_TEST(WithNanTest) {
        TJsonReaderConfig cfg;
        cfg.AllowReadNanInf = true;

        TJsonValue out;
        // This read will ok
        UNIT_ASSERT(ReadJsonTree(JSON_NAN_TEST, &cfg, &out, /* throwOnError */ false));
    }
}

