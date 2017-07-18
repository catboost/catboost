#include "hex.h"

#include <library/unittest/registar.h>
#include "str.h"

SIMPLE_UNIT_TEST_SUITE(THexCodingTest) {
    void TestImpl(const TString& data) {
        TString encoded;
        TStringOutput encodedOut(encoded);
        HexEncode(data.Data(), data.Size(), encodedOut);

        UNIT_ASSERT_EQUAL(encoded.Size(), data.Size() * 2);

        TString decoded;
        TStringOutput decodedOut(decoded);
        HexDecode(encoded.Data(), encoded.Size(), decodedOut);

        UNIT_ASSERT_EQUAL(decoded, data);
    }

    SIMPLE_UNIT_TEST(TestEncodeDecodeToStream) {
        TString data = "100ABAcaba500,$%0987123456   \n\t\x01\x02\x03.";
        TestImpl(data);
    }

    SIMPLE_UNIT_TEST(TestEmpty) {
        TestImpl("");
    }
}
