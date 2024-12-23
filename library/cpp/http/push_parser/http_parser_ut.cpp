#include "http_parser.h"

#include <library/cpp/testing/unittest/registar.h>
#include <library/cpp/string_utils/base64/base64.h>

#include <util/stream/str.h>
#include <util/stream/zlib.h>

namespace {
    template <size_t N>
    bool Parse(THttpParser& p, const char (&data)[N]) {
        return p.Parse(data, N - 1);
    }

    TString MakeEncodedRequest(const TString& encoding, const TString& data) {
        TStringStream msg;
        msg << "POST / HTTP/1.1\r\n"
               "Content-Encoding: "
            << encoding << " \r\n"
                           "Content-Length: "
            << data.size() << "\r\n\r\n"
            << data;
        return msg.Str();
    }

    TString MakeEncodedResponse(const TString& encoding, const TString& data) {
        TStringStream msg;
        msg << "HTTP/1.1 200\r\n"
               "Content-Encoding: "
            << encoding << " \r\n"
                           "Content-Length: "
            << data.size() << "\r\n\r\n"
            << data;
        return msg.Str();
    }
}

Y_UNIT_TEST_SUITE(THttpParser) {
    Y_UNIT_TEST(TParsingToEof) {
        {
            THttpParser p(THttpParser::Request);
            UNIT_ASSERT(!Parse(p, "GET "));
            UNIT_ASSERT(!Parse(p, "/test/test?text=123 HTTP/1.0\r"));
            UNIT_ASSERT(!Parse(p, "\nHost: yabs.yandex.ru"));
            UNIT_ASSERT(!Parse(p, "\r\nAccept-eNcoding: *\r\n"));
            UNIT_ASSERT(!Parse(p, "Accept-eNcoding:\r\n"));
            UNIT_ASSERT(!Parse(p, "Accept-eNcoding: , ,\r\n"));
            UNIT_ASSERT(Parse(p, "\r\n"));

            UNIT_ASSERT_VALUES_EQUAL(p.IsKeepAlive(), false);
            UNIT_ASSERT_VALUES_EQUAL(p.GetBestCompressionScheme(), "gzip");
            THashSet<TString> acceptedEncodings = {"*"};
            UNIT_ASSERT_VALUES_EQUAL(p.AcceptedEncodings(), acceptedEncodings);
            ui64 cl;
            UNIT_ASSERT_VALUES_EQUAL(p.GetContentLength(cl), false);
        }
        {
            THttpParser p(THttpParser::Request);
            UNIT_ASSERT(!Parse(p, "GET"));
            UNIT_ASSERT(!Parse(p, " /test/test?text=123 HTTP/1.1\r"));
            UNIT_ASSERT(!Parse(p, "\nHost: yabs.yandex.ru"));
            UNIT_ASSERT(!Parse(p, "\r\nAccept-eNcoding: yyy123\r\n"));
            UNIT_ASSERT(!Parse(p, "Accept-eNcoding: aaa\r\n"));
            UNIT_ASSERT(!Parse(p, "Accept-eNcoding: y-LZQ, y-Lzo , bbb\r\n"));
            UNIT_ASSERT(!Parse(p, "accept-encoding:ccc\r\n"));
            UNIT_ASSERT(Parse(p, "\r\n"));

            UNIT_ASSERT_VALUES_EQUAL(p.IsKeepAlive(), true);
            UNIT_ASSERT_VALUES_EQUAL(p.GetBestCompressionScheme(), "y-lzo");
            THashSet<TString> acceptedEncodings = {"yyy123", "aaa", "y-lzq", "y-lzo", "bbb", "ccc"};
            UNIT_ASSERT_VALUES_EQUAL(p.AcceptedEncodings(), acceptedEncodings);
            ui64 cl;
            UNIT_ASSERT_VALUES_EQUAL(p.GetContentLength(cl), false);
        }
        {
            THttpParser p(THttpParser::Request);
            UNIT_ASSERT(!Parse(p, "GET /"));
            UNIT_ASSERT(!Parse(p, "test/test?text=123 HTTP/1."));
            UNIT_ASSERT(Parse(p, "0\r\nHost: yabs.yandex.ru\r\n\r\n"));

            UNIT_ASSERT_VALUES_EQUAL(p.IsKeepAlive(), false);
            UNIT_ASSERT_VALUES_EQUAL(p.GetBestCompressionScheme(), "");
            ui64 cl;
            UNIT_ASSERT_VALUES_EQUAL(p.GetContentLength(cl), false);
        }
        {
            THttpParser p;
            UNIT_ASSERT(!Parse(p, "HT"));
            UNIT_ASSERT(!Parse(p, "TP/1.0 200 OK\r"));
            UNIT_ASSERT(!Parse(p, "\nContent-Type: text/plain; charset=utf-8\r\n"
                                  "\r\n"));
            UNIT_ASSERT(Parse(p, ""));

            UNIT_ASSERT_VALUES_EQUAL(p.RetCode(), 200u);
            UNIT_ASSERT_VALUES_EQUAL(p.IsKeepAlive(), false);
            ui64 cl;
            UNIT_ASSERT_VALUES_EQUAL(p.GetContentLength(cl), false);
        }
        {
            THttpParser p;
            UNIT_ASSERT(!Parse(p, "H"));
            UNIT_ASSERT(!Parse(p, "TTP/1.1 200 OK\r"));
            UNIT_ASSERT(!Parse(p, "\n"));
            UNIT_ASSERT(!Parse(p, "Content-Type: text/plain; charset=utf-8\r\n"
                                  "\r\n"));
            UNIT_ASSERT(Parse(p, ""));

            UNIT_ASSERT_VALUES_EQUAL(p.RetCode(), 200u);
            UNIT_ASSERT_VALUES_EQUAL(p.IsKeepAlive(), true);
            ui64 cl;
            UNIT_ASSERT_VALUES_EQUAL(p.GetContentLength(cl), false);
        }
    }

    Y_UNIT_TEST(TParsingToContentLength) {
        THttpParser p;
        UNIT_ASSERT(!Parse(p, "HTTP/1.1 404 OK"));
        UNIT_ASSERT(!Parse(p, "\r\nConnection: close\r"));
        UNIT_ASSERT(!Parse(p, "\nContent-Length: 10\r"));
        UNIT_ASSERT(Parse(p, "\n\r\n0123456789"));

        UNIT_ASSERT_VALUES_EQUAL(p.RetCode(), 404u);
        UNIT_ASSERT_VALUES_EQUAL(p.IsKeepAlive(), false);
        ui64 cl = 0;
        UNIT_ASSERT_VALUES_EQUAL(p.GetContentLength(cl), true);
        UNIT_ASSERT_VALUES_EQUAL(cl, 10u);
        UNIT_ASSERT_VALUES_EQUAL(p.Content(), "0123456789");
    }

    Y_UNIT_TEST(TParsingContentLengthZero) {
        THttpParser p(THttpParser::Request);
        UNIT_ASSERT(!Parse(p, "GET /test HTTP/1.1"));
        UNIT_ASSERT(!Parse(p, "\r\nAccept: *\r"));
        UNIT_ASSERT(!Parse(p, "\nContent-Length: 0\r\n"));
        UNIT_ASSERT(!Parse(p, "Cookie: a=b\r\nX-Foo: "));
        UNIT_ASSERT(!Parse(p, "bar\r\nX-Bar: foo\r"));
        UNIT_ASSERT(Parse(p, "\n\r\n"));

        auto ch = p.Headers().FindHeader("Cookie");
        UNIT_ASSERT(ch);
        UNIT_ASSERT_VALUES_EQUAL(ch->Value(), "a=b");

        auto xh = p.Headers().FindHeader("X-Foo");
        UNIT_ASSERT(xh);
        UNIT_ASSERT_VALUES_EQUAL(xh->Value(), "bar");

        auto bh = p.Headers().FindHeader("X-Bar");
        UNIT_ASSERT(bh);
        UNIT_ASSERT_VALUES_EQUAL(bh->Value(), "foo");
    }

    Y_UNIT_TEST(TParsingChunkedContent) {
        {
            THttpParser p;
            UNIT_ASSERT(!Parse(p, "HTTP/1.1 333 OK\r\nC"));
            UNIT_ASSERT(!Parse(p, "onnection: Keep-Alive\r\n"));
            UNIT_ASSERT(!Parse(p, "Transfer-Encoding: chunked\r\n\r\n"));
            UNIT_ASSERT(Parse(p, "8\r\n01234567\r\n0\r\n\r\n---"));

            UNIT_ASSERT_VALUES_EQUAL(p.RetCode(), 333u);
            UNIT_ASSERT_VALUES_EQUAL(p.IsKeepAlive(), true);
            ui64 cl = 0;
            UNIT_ASSERT_VALUES_EQUAL(p.GetContentLength(cl), false);
            UNIT_ASSERT_VALUES_EQUAL(p.Content(), "01234567");

            THttpHeaders::TConstIterator it = p.Headers().Begin();
            UNIT_ASSERT_VALUES_EQUAL(it->ToString(), TString("Connection: Keep-Alive"));
            UNIT_ASSERT_VALUES_EQUAL((++it)->ToString(), TString("Transfer-Encoding: chunked"));

            UNIT_ASSERT_VALUES_EQUAL(p.GetExtraDataSize(), 3u);
        }
        {
            //parse by tiny blocks (1 byte)
            THttpParser p;
            const char msg[] = "HTTP/1.1 333 OK\r\n"
                               "Connection: Keep-Alive\r\n"
                               "Transfer-Encoding: chunked\r\n\r\n"
                               "8 ; key=value \r\n01234567\r\n"
                               "0\r\n\r\n";

            for (size_t i = 0; i < (sizeof(msg) - 2); ++i) {
                UNIT_ASSERT(!p.Parse(msg + i, 1));
            }
            UNIT_ASSERT(p.Parse(msg + sizeof(msg) - 2, 1));

            UNIT_ASSERT_VALUES_EQUAL(p.RetCode(), 333u);
            UNIT_ASSERT_VALUES_EQUAL(p.IsKeepAlive(), true);
            ui64 cl = 0;
            UNIT_ASSERT_VALUES_EQUAL(p.GetContentLength(cl), false);
            UNIT_ASSERT_VALUES_EQUAL(p.Content(), "01234567");

            THttpHeaders::TConstIterator it = p.Headers().Begin();
            UNIT_ASSERT_VALUES_EQUAL(it->ToString(), TString("Connection: Keep-Alive"));
            UNIT_ASSERT_VALUES_EQUAL((++it)->ToString(), TString("Transfer-Encoding: chunked"));

            THttpParser p2;
            UNIT_ASSERT(!p2.Parse(msg, sizeof(msg) - 2));
            UNIT_ASSERT(p2.Parse(msg + sizeof(msg) - 2, 1));
        }
        {
            //lot of chunks
            THttpParser p;

            TStringStream msgStream;
            msgStream << "HTTP/1.1 333 OK\r\n"
                   "Connection: Keep-Alive\r\n"
                   "Transfer-Encoding: chunked\r\n\r\n";

            const size_t chunkCount = 100;

            TVector<size_t> partSizes(chunkCount + 1);
            TString content;

            partSizes[0] = msgStream.Str().size();
            for (size_t i = 0; i < chunkCount; ++i) {
                const TString chunk = NUnitTest::RandomString(i + 1);
                content += chunk;

                const size_t prevPos = msgStream.Str().size();
                msgStream << Hex(chunk.size(), 0) << "\r\n" << chunk << "\r\n";
                partSizes[i + 1] = (msgStream.Str().size() - prevPos);
            }

            msgStream << "0\r\n\r\n";

            const TStringBuf msg(msgStream.Str());

            const auto check = [content](const THttpParser& p) {
                UNIT_ASSERT_VALUES_EQUAL(p.RetCode(), 333u);
                UNIT_ASSERT_VALUES_EQUAL(p.IsKeepAlive(), true);
                ui64 cl = 0;
                UNIT_ASSERT_VALUES_EQUAL(p.GetContentLength(cl), false);
                UNIT_ASSERT_VALUES_EQUAL(p.Content(), content);

                THttpHeaders::TConstIterator it = p.Headers().Begin();
                UNIT_ASSERT_VALUES_EQUAL(it->ToString(), TString("Connection: Keep-Alive"));
                UNIT_ASSERT_VALUES_EQUAL((++it)->ToString(), TString("Transfer-Encoding: chunked"));
            };

            {//1 byte
                THttpParser p;

                for (size_t i = 0; i < (msg.size() - 1); ++i) {
                    UNIT_ASSERT(!p.Parse(msg.data() + i, 1));
                }

                UNIT_ASSERT(p.Parse(msg.data() + msg.size() - 1, 1));

                check(p);
            }

            {//by chunk
                size_t pos = 0;
                THttpParser p;

                for (size_t i = 0; i < partSizes.size(); ++i) {
                    UNIT_ASSERT(!p.Parse(msg.data() + pos, partSizes[i]));
                    pos += partSizes[i];
                }

                UNIT_ASSERT(p.Parse(msg.data() + pos, msg.size() - pos));

                check(p);
            }

            {//two parts
                for (size_t i = 1; i < msg.size(); ++i) {
                    THttpParser p;
                    UNIT_ASSERT(!p.Parse(msg.data(), i));
                    UNIT_ASSERT(p.Parse(msg.data() + i, msg.size() - i));

                    check(p);
                }
            }
        }
    }

    Y_UNIT_TEST(TParsingEncodedContent) {
        /// parse request with encoded content
        TString testLine = "test line";
        {
            // test identity
            THttpParser p(THttpParser::Request);
            TString msg = MakeEncodedRequest("identity", testLine);
            UNIT_ASSERT(p.Parse(msg.data(), msg.size()));
            UNIT_ASSERT_VALUES_EQUAL(p.DecodedContent(), testLine);
        }
        {
            // test deflate
            THttpParser p(THttpParser::Request);
            TString zlibTestLine = "\x78\x9C\x2B\x49\x2D\x2E\x51\xC8\xC9\xCC\x4B\x05\x00\x11\xEE\x03\x89";
            TString msg = MakeEncodedRequest("deflate", zlibTestLine);
            UNIT_ASSERT(p.Parse(msg.data(), msg.size()));
            UNIT_ASSERT_VALUES_EQUAL(p.DecodedContent(), testLine);
        }
        {
            // test gzip
            THttpParser p(THttpParser::Request);
            TString gzipTestLine(
                "\x1f\x8b\x08\x08\x5e\xdd\xa8\x56\x00\x03\x74\x6c\x00\x2b\x49\x2d"
                "\x2e\x51\xc8\xc9\xcc\x4b\x05\x00\x27\xe9\xef\xaf\x09\x00\x00\x00"sv);
            TString msg = MakeEncodedRequest("gzip", gzipTestLine);
            UNIT_ASSERT(p.Parse(msg.data(), msg.size()));
            UNIT_ASSERT_VALUES_EQUAL(p.DecodedContent(), testLine);
        }
        {
            // test snappy
            THttpParser p(THttpParser::Request);
            TString snappyTestLine(
                "*\xc7\x10\x00\x00\x00\x00\x00\x00\x00\x0e"
                "42.230-20181121*\xc7\x01\x00\x00\x00\x00\x00\x00\x00\x00"sv);
            TString msg = MakeEncodedRequest("z-snappy", snappyTestLine);
            UNIT_ASSERT(p.Parse(msg.data(), msg.size()));
            UNIT_ASSERT_VALUES_EQUAL(p.DecodedContent(), "2.230-20181121");
        }
        {
            // test unknown compressor
            THttpParser p(THttpParser::Request);
            TString content = "some trash";
            TString msg = MakeEncodedRequest("unknown", content);
            UNIT_ASSERT_EXCEPTION(p.Parse(msg.data(), msg.size()), THttpParseException);
        }
        {
            for (auto contentEncoding : TVector<TString>{"z-unknown", "z-zstd06", "z-zstd08", "z-zstd08-0"}) {
                // test unknown blockcodec compressor
                THttpParser p(THttpParser::Request);
                TString content = "some trash";
                TString msg = MakeEncodedRequest(contentEncoding, content);
                UNIT_ASSERT_EXCEPTION(p.Parse(msg.data(), msg.size()), THttpParseException);
            }
        }
        {
            // test broken deflate
            THttpParser p(THttpParser::Request);
            TString content(TStringBuf("some trash ....................."));
            TString msg = MakeEncodedRequest("deflate", content);
            UNIT_ASSERT_EXCEPTION(p.Parse(msg.data(), msg.size()), yexception);
        }
        {
            // test broken gzip
            THttpParser p(THttpParser::Request);
            TString content(
                "\x1f\x8b\x08\x08\x5e\xdd\xa8\x56\x00\x03\x74\x6c\x00\x2b\x49\x2d"
                "\x2e\x51\xc8\xc9\xcc\x4b\x05\x00\x27\xe9\xef\xaf\x09some trash\x00\x00\x00"sv);
            TString msg = MakeEncodedRequest("gzip", content);
            UNIT_ASSERT_EXCEPTION(p.Parse(msg.data(), msg.size()), yexception);
        }
        {
            // test broken snappy
            THttpParser p(THttpParser::Request);
            TString snappyTestLine(TStringBuf("\x1b some very\x05,long payload"sv));
            TString msg = MakeEncodedRequest("z-snappy", snappyTestLine);
            UNIT_ASSERT_EXCEPTION(p.Parse(msg.data(), msg.size()), yexception);
        }
        {
            // raw content

            const TString testBody = "lalalabububu";
            THttpParser p(THttpParser::Request);
            TString content;
            {
                TStringOutput output(content);
                TZLibCompress compress(&output, ZLib::Raw);
                compress.Write(testBody.data(), testBody.size());
            }

            TString msg = MakeEncodedRequest("deflate", content);
            UNIT_ASSERT(p.Parse(msg.data(), msg.size()));
            UNIT_ASSERT_VALUES_EQUAL(p.DecodedContent(), testBody);
        }

        {
            // test gzip response
            THttpParser p(THttpParser::Response);
            TString gzipTestLine(
                "\x1f\x8b\x08\x08\x5e\xdd\xa8\x56\x00\x03\x74\x6c\x00\x2b\x49\x2d"
                "\x2e\x51\xc8\xc9\xcc\x4b\x05\x00\x27\xe9\xef\xaf\x09\x00\x00\x00"sv);
            TString msg = MakeEncodedResponse("gzip", gzipTestLine);
            UNIT_ASSERT(p.Parse(msg.data(), msg.size()));
            UNIT_ASSERT_VALUES_EQUAL(p.DecodedContent(), testLine);
        }
        {
            // test br response
            THttpParser p(THttpParser::Response);
            TString brTestLine = Base64Decode("CwSAdGVzdCBsaW5lAw==");
            TString msg = MakeEncodedResponse("br", brTestLine);
            UNIT_ASSERT(p.Parse(msg.data(), msg.size()));
            UNIT_ASSERT_VALUES_EQUAL(p.DecodedContent(), testLine);
        }
        {
            // test gzip response with trailing garbage
            THttpParser p(THttpParser::Response);
            p.SetGzipAllowMultipleStreams(false);
            TString gzipTestLine(
                "\x1f\x8b\x08\x08\x5e\xdd\xa8\x56\x00\x03\x74\x6c\x00\x2b\x49\x2d"
                "\x2e\x51\xc8\xc9\xcc\x4b\x05\x00\x27\xe9\xef\xaf\x09\x00\x00\x00garbage"sv);
            TString msg = MakeEncodedResponse("gzip", gzipTestLine);
            UNIT_ASSERT(p.Parse(msg.data(), msg.size()));
            UNIT_ASSERT_VALUES_EQUAL(p.DecodedContent(), testLine);
        }
        {
            // test gzip response with trailing garbage exception
            THttpParser p(THttpParser::Response);
            TString gzipTestLine(
                "\x1f\x8b\x08\x08\x5e\xdd\xa8\x56\x00\x03\x74\x6c\x00\x2b\x49\x2d"
                "\x2e\x51\xc8\xc9\xcc\x4b\x05\x00\x27\xe9\xef\xaf\x09\x00\x00\x00garbage"sv);
            TString msg = MakeEncodedResponse("gzip", gzipTestLine);
            UNIT_ASSERT_EXCEPTION(p.Parse(msg.data(), msg.size()), yexception);
        }
        {
            // test br garbage response
            THttpParser p(THttpParser::Response);
            TString brTestLine(
                "\x1f\x8b\x08\x08\x5e\xdd\xa8\x56\x00\x03\x74\x6c\x00\x2b\x49\x2d"
                "\x2e\x51\xc8\xc9\xcc\x4b\x05\x00\x27\xe9\xef\xaf\x09\x00\x00\x00"sv);
            TString msg = MakeEncodedResponse("br", brTestLine);
            UNIT_ASSERT_EXCEPTION(p.Parse(msg.data(), msg.size()), yexception);
        }
        {
            // test disable decode content
            THttpParser p(THttpParser::Response);
            p.DisableDecodeContent();
            TString gzipTestLine(
                "\x1f\x8b\x08\x08\x5e\xdd\xa8\x56\x00\x03\x74\x6c\x00\x2b\x49\x2d"
                "\x2e\x51\xc8\xc9\xcc\x4b\x05\x00\x27\xe9\xef\xaf\x09\x00\x00\x00"sv);
            TString msg = MakeEncodedResponse("gzip", gzipTestLine);
            UNIT_ASSERT(p.Parse(msg.data(), msg.size()));
            UNIT_ASSERT_VALUES_EQUAL(p.DecodedContent(), "");
            UNIT_ASSERT_VALUES_EQUAL(p.Content(), gzipTestLine);
        }
        {
            // test manual content decoding
            THttpParser p(THttpParser::Response);
            p.DisableDecodeContent();
            TString gzipTestLine(
                "\x1f\x8b\x08\x08\x5e\xdd\xa8\x56\x00\x03\x74\x6c\x00\x2b\x49\x2d"
                "\x2e\x51\xc8\xc9\xcc\x4b\x05\x00\x27\xe9\xef\xaf\x09\x00\x00\x00"sv);
            TString msg = MakeEncodedResponse("gzip", gzipTestLine);
            UNIT_ASSERT(p.Parse(msg.data(), msg.size()));
            UNIT_ASSERT_VALUES_EQUAL(p.DecodedContent(), "");

            TString decoded;
            UNIT_ASSERT(p.DecodeContent(decoded));
            UNIT_ASSERT_VALUES_EQUAL(decoded, testLine);
        }
    }

    Y_UNIT_TEST(TParsingMultilineHeaders) {
        THttpParser p;
        UNIT_ASSERT(!Parse(p, "HTTP/1.1 444 OK\r\n"));
        UNIT_ASSERT(!Parse(p, "Vary: Accept-Encoding, \r\n"));
        UNIT_ASSERT(!Parse(p, "\tAccept-Language\r\n"
                              "Host: \tany.com\t \r\n\r\n"
                              "01234567"));
        UNIT_ASSERT(Parse(p, ""));

        UNIT_ASSERT_VALUES_EQUAL(p.Content(), "01234567");

        THttpHeaders::TConstIterator it = p.Headers().Begin();
        UNIT_ASSERT_VALUES_EQUAL(it->ToString(), TString("Vary: Accept-Encoding, \tAccept-Language"));
        UNIT_ASSERT_VALUES_EQUAL((++it)->ToString(), TString("Host: any.com"));
    }

    Y_UNIT_TEST(THttpIoStreamInteroperability) {
        TStringBuf content = "very very very long content";

        TMemoryInput request("GET / HTTP/1.1\r\nAccept-Encoding: z-snappy\r\n\r\n");
        THttpInput i(&request);

        TString result;
        TStringOutput out(result);
        THttpOutput httpOut(&out, &i);
        httpOut.EnableCompression(true);
        httpOut << "HTTP/1.1 200 OK\r\n";
        httpOut << "Content-Length: " << content.size() << "\r\n\r\n";
        httpOut << content;
        httpOut.Finish();
        // check that compression works
        UNIT_ASSERT(!result.Contains(content));

        THttpParser p;
        UNIT_ASSERT(p.Parse(result.data(), result.size()));
        UNIT_ASSERT_VALUES_EQUAL(p.RetCode(), 200);
        UNIT_ASSERT(p.Headers().HasHeader("Content-Encoding"));
        UNIT_ASSERT_VALUES_EQUAL(p.DecodedContent(), content);
    }

    Y_UNIT_TEST(TBodyNotExpected) {
        THttpParser p;
        p.BodyNotExpected();
        UNIT_ASSERT(!Parse(p, "HTTP/1.1 200 OK\r\n"));
        UNIT_ASSERT(!Parse(p, "Content-Length: 1234\r\n"));
        UNIT_ASSERT(Parse(p, "\r\n"));

        UNIT_ASSERT(p.Headers().HasHeader("Content-Length"));
        UNIT_ASSERT(p.Headers().FindHeader("Content-Length")->Value() == "1234");
    }

    Y_UNIT_TEST(TBodyIgnoredIfNotExpected) {
        THttpParser p;
        p.BodyNotExpected();
        UNIT_ASSERT(!Parse(p, "HTTP/1.1 200 OK\r\n"));
        UNIT_ASSERT(!Parse(p, "Content-Length: 5\r\n"));
        UNIT_ASSERT(Parse(p, "\r\nHello"));

        UNIT_ASSERT(p.Content().empty());
        UNIT_ASSERT(p.GetExtraDataSize() == 5);
    }
}
