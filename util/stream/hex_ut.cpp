#include "hex.h"

#include <library/cpp/testing/unittest/registar.h>
#include "str.h"

Y_UNIT_TEST_SUITE(THexCodingTest) {
    void TestImpl(const TString& data) {
        TString encoded;
        TStringOutput encodedOut(encoded);
        HexEncode(data.data(), data.size(), encodedOut);

        UNIT_ASSERT_EQUAL(encoded.size(), data.size() * 2);

        TString decoded;
        TStringOutput decodedOut(decoded);
        HexDecode(encoded.data(), encoded.size(), decodedOut);

        UNIT_ASSERT_EQUAL(decoded, data);
    }

    Y_UNIT_TEST(TestEncodeDecodeToStream) {
        TString data = "100ABAcaba500,$%0987123456   \n\t\x01\x02\x03.";
        TestImpl(data);
    }

    Y_UNIT_TEST(TestEmpty) {
        TestImpl("");
    }
} // Y_UNIT_TEST_SUITE(THexCodingTest)
