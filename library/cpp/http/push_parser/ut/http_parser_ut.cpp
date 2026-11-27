#include <library/cpp/http/push_parser/http_parser.h>
#include <library/cpp/testing/unittest/registar.h>

Y_UNIT_TEST_SUITE(THttpParserEofTest) {
    Y_UNIT_TEST(PrematureEof) {
        for (TStringBuf partial : {
                 TStringBuf(""),
                 TStringBuf("HTTP/1.1 20"),
                 TStringBuf("HTTP/1.1 200 OK\r\nContent-Length: 4\r\n"),
                 TStringBuf("HTTP/1.1 200 OK\r\nContent-Length: 4\r\n\r\nhi"),
                 TStringBuf("HTTP/1.1 200 OK\r\nTransfer-Encoding: chunked\r\n\r\n2\r\nhi\r\n"),
             }) {
            THttpParser parser;
            if (partial) {
                UNIT_ASSERT(!parser.Parse(partial.data(), partial.size()));
            }
            UNIT_ASSERT(!parser.TryFinishOnEof());
            UNIT_ASSERT_EXCEPTION(parser.Parse("", 0), THttpException);
        }
    }

    Y_UNIT_TEST(CloseDelimitedResponse) {
        constexpr TStringBuf response = "HTTP/1.1 200 OK\r\nConnection: close\r\n\r\nbody";
        for (bool useStatus : {false, true}) {
            THttpParser parser;
            UNIT_ASSERT(!parser.Parse(response.data(), response.size()));
            UNIT_ASSERT(useStatus ? parser.TryFinishOnEof() : parser.Parse("", 0));
            UNIT_ASSERT_VALUES_EQUAL(parser.RetCode(), 200);
            UNIT_ASSERT_VALUES_EQUAL(parser.DecodedContent(), "body");
        }
    }
}
