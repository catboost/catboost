#include "stream.h"
#include "chunk.h"

#include <library/cpp/http/server/http_ex.h>

#include <library/cpp/testing/unittest/registar.h>
#include <library/cpp/testing/unittest/tests_data.h>

#include <util/string/printf.h>
#include <util/network/socket.h>
#include <util/stream/file.h>
#include <util/stream/output.h>
#include <util/stream/tee.h>
#include <util/stream/zlib.h>
#include <util/stream/null.h>

Y_UNIT_TEST_SUITE(THttpStreamTest) {
    class TTestHttpServer: public THttpServer::ICallBack {
        class TRequest: public THttpClientRequestEx {
        public:
            inline TRequest(TTestHttpServer* parent)
                : Parent_(parent)
            {
            }

            bool Reply(void* /*tsr*/) override {
                if (!ProcessHeaders()) {
                    return true;
                }

                // Check that function will not hang on
                Input().ReadAll();

                // "lo" is for "local"
                if (RD.ServerName() == "yandex.lo") {
                    // do redirect
                    Output() << "HTTP/1.1 301 Moved permanently\r\n"
                                "Location: http://www.yandex.lo\r\n"
                                "\r\n";
                } else if (RD.ServerName() == "www.yandex.lo") {
                    Output() << "HTTP/1.1 200 Ok\r\n"
                                "\r\n";
                } else {
                    Output() << "HTTP/1.1 200 Ok\r\n\r\n";
                    if (Buf.Size()) {
                        Output().Write(Buf.AsCharPtr(), Buf.Size());
                    } else {
                        Output() << Parent_->Res_;
                    }
                }
                Output().Finish();

                Parent_->LastRequestSentSize_ = Output().SentSize();

                return true;
            }

        private:
            TTestHttpServer* Parent_ = nullptr;
        };

    public:
        inline TTestHttpServer(const TString& res)
            : Res_(res)
        {
        }

        TClientRequest* CreateClient() override {
            return new TRequest(this);
        }

        size_t LastRequestSentSize() const {
            return LastRequestSentSize_;
        }

    private:
        TString Res_;
        size_t LastRequestSentSize_ = 0;
    };

    Y_UNIT_TEST(TestCodings1) {
        UNIT_ASSERT(SupportedCodings().size() > 0);
    }

    Y_UNIT_TEST(TestHttpInput) {
        TString res = "I'm a teapot";
        TPortManager pm;
        const ui16 port = pm.GetPort();

        TTestHttpServer serverImpl(res);
        THttpServer server(&serverImpl, THttpServer::TOptions(port).EnableKeepAlive(true).EnableCompression(true));

        UNIT_ASSERT(server.Start());

        TNetworkAddress addr("localhost", port);
        TSocket s(addr);

        //TDebugOutput dbg;
        TNullOutput dbg;

        {
            TSocketOutput so(s);
            TTeeOutput out(&so, &dbg);
            THttpOutput output(&out);

            output.EnableKeepAlive(true);
            output.EnableCompression(true);

            TString r;
            r += "GET / HTTP/1.1";
            r += "\r\n";
            r += "Host: yandex.lo";
            r += "\r\n";
            r += "\r\n";

            output.Write(r.data(), r.size());
            output.Finish();
        }

        {
            TSocketInput si(s);
            THttpInput input(&si);
            unsigned httpCode = ParseHttpRetCode(input.FirstLine());
            UNIT_ASSERT_VALUES_EQUAL(httpCode / 10, 30u);

            TransferData(&input, &dbg);
        }
        server.Stop();
    }

    Y_UNIT_TEST(TestHttpInputDelete) {
        TString res = "I'm a teapot";
        TPortManager pm;
        const ui16 port = pm.GetPort();

        TTestHttpServer serverImpl(res);
        THttpServer server(&serverImpl, THttpServer::TOptions(port).EnableKeepAlive(true).EnableCompression(true));

        UNIT_ASSERT(server.Start());

        TNetworkAddress addr("localhost", port);
        TSocket s(addr);

        //TDebugOutput dbg;
        TNullOutput dbg;

        {
            TSocketOutput so(s);
            TTeeOutput out(&so, &dbg);
            THttpOutput output(&out);

            output.EnableKeepAlive(true);
            output.EnableCompression(true);

            TString r;
            r += "DELETE / HTTP/1.1";
            r += "\r\n";
            r += "Host: yandex.lo";
            r += "\r\n";
            r += "\r\n";

            output.Write(r.data(), r.size());
            output.Finish();
        }

        {
            TSocketInput si(s);
            THttpInput input(&si);
            unsigned httpCode = ParseHttpRetCode(input.FirstLine());
            UNIT_ASSERT_VALUES_EQUAL(httpCode / 10, 30u);

            TransferData(&input, &dbg);
        }
        server.Stop();
    }

    Y_UNIT_TEST(TestParseHttpRetCode) {
        UNIT_ASSERT_VALUES_EQUAL(ParseHttpRetCode("HTTP/1.1 301"), 301u);
    }

    Y_UNIT_TEST(TestKeepAlive) {
        {
            TString s = "GET / HTTP/1.0\r\n\r\n";
            TStringInput si(s);
            THttpInput in(&si);
            UNIT_ASSERT(!in.IsKeepAlive());
        }

        {
            TString s = "GET / HTTP/1.0\r\nConnection: keep-alive\r\n\r\n";
            TStringInput si(s);
            THttpInput in(&si);
            UNIT_ASSERT(in.IsKeepAlive());
        }

        {
            TString s = "GET / HTTP/1.1\r\n\r\n";
            TStringInput si(s);
            THttpInput in(&si);
            UNIT_ASSERT(in.IsKeepAlive());
        }

        {
            TString s = "GET / HTTP/1.1\r\nConnection: close\r\n\r\n";
            TStringInput si(s);
            THttpInput in(&si);
            UNIT_ASSERT(!in.IsKeepAlive());
        }

        {
            TString s = "HTTP/1.0 200 Ok\r\n\r\n";
            TStringInput si(s);
            THttpInput in(&si);
            UNIT_ASSERT(!in.IsKeepAlive());
        }

        {
            TString s = "HTTP/1.0 200 Ok\r\nConnection: keep-alive\r\n\r\n";
            TStringInput si(s);
            THttpInput in(&si);
            UNIT_ASSERT(in.IsKeepAlive());
        }

        {
            TString s = "HTTP/1.1 200 Ok\r\n\r\n";
            TStringInput si(s);
            THttpInput in(&si);
            UNIT_ASSERT(in.IsKeepAlive());
        }

        {
            TString s = "HTTP/1.1 200 Ok\r\nConnection: close\r\n\r\n";
            TStringInput si(s);
            THttpInput in(&si);
            UNIT_ASSERT(!in.IsKeepAlive());
        }
    }

    Y_UNIT_TEST(TestMinRequest) {
        TString res = "qqqqqq";
        TPortManager pm;
        const ui16 port = pm.GetPort();

        TTestHttpServer serverImpl(res);
        THttpServer server(&serverImpl, THttpServer::TOptions(port).EnableKeepAlive(true).EnableCompression(true));

        UNIT_ASSERT(server.Start());

        TNetworkAddress addr("localhost", port);

        TSocket s(addr);
        TNullOutput dbg;

        SendMinimalHttpRequest(s, "www.yandex.lo", "/");

        TSocketInput si(s);
        THttpInput input(&si);
        unsigned httpCode = ParseHttpRetCode(input.FirstLine());
        UNIT_ASSERT_VALUES_EQUAL(httpCode, 200u);

        TransferData(&input, &dbg);
        server.Stop();
    }

    Y_UNIT_TEST(TestResponseWithBlanks) {
        TString res = "qqqqqq\r\n\r\nsdasdsad\r\n";
        TPortManager pm;
        const ui16 port = pm.GetPort();

        TTestHttpServer serverImpl(res);
        THttpServer server(&serverImpl, THttpServer::TOptions(port).EnableKeepAlive(true).EnableCompression(true));

        UNIT_ASSERT(server.Start());

        TNetworkAddress addr("localhost", port);

        TSocket s(addr);

        SendMinimalHttpRequest(s, "www.yandex.ru", "/");

        TSocketInput si(s);
        THttpInput input(&si);
        unsigned httpCode = ParseHttpRetCode(input.FirstLine());
        UNIT_ASSERT_VALUES_EQUAL(httpCode, 200u);
        TString reply = input.ReadAll();
        UNIT_ASSERT_VALUES_EQUAL(reply, res);
        server.Stop();
    }

    Y_UNIT_TEST(TestOutputFlush) {
        TString str;
        TStringOutput strOut(str);
        TBufferedOutput bufOut(&strOut, 8192);
        THttpOutput httpOut(&bufOut);

        httpOut.EnableKeepAlive(true);
        httpOut.EnableCompression(true);

        const char* header = "GET / HTTP/1.1\r\nHost: yandex.ru\r\n\r\n";
        httpOut << header;

        unsigned curLen = str.size();
        const char* body = "<html>Hello</html>";
        httpOut << body;
        UNIT_ASSERT_VALUES_EQUAL(curLen, str.size());
        httpOut.Flush();
        UNIT_ASSERT_VALUES_EQUAL(curLen + strlen(body), str.size());
    }

    Y_UNIT_TEST(TestOutputPostFlush) {
        TString str;
        TString checkStr;
        TStringOutput strOut(str);
        TStringOutput checkOut(checkStr);
        TBufferedOutput bufOut(&strOut, 8192);
        TTeeOutput teeOut(&bufOut, &checkOut);
        THttpOutput httpOut(&teeOut);

        httpOut.EnableKeepAlive(true);
        httpOut.EnableCompression(true);

        const char* header = "POST / HTTP/1.1\r\nHost: yandex.ru\r\n\r\n";
        httpOut << header;

        UNIT_ASSERT_VALUES_EQUAL(str.size(), 0u);

        const char* body = "<html>Hello</html>";
        httpOut << body;
        UNIT_ASSERT_VALUES_EQUAL(str.size(), 0u);

        httpOut.Flush();
        UNIT_ASSERT_VALUES_EQUAL(checkStr.size(), str.size());
    }

    TString MakeHttpOutputBody(const char* body, bool encodingEnabled) {
        TString str;
        TStringOutput strOut(str);
        {
            TBufferedOutput bufOut(&strOut, 8192);
            THttpOutput httpOut(&bufOut);

            httpOut.EnableKeepAlive(true);
            httpOut.EnableCompression(true);
            httpOut.EnableBodyEncoding(encodingEnabled);

            httpOut << "POST / HTTP/1.1\r\n";
            httpOut << "Host: yandex.ru\r\n";
            httpOut << "Content-Encoding: gzip\r\n";
            httpOut << "\r\n";

            UNIT_ASSERT_VALUES_EQUAL(str.size(), 0u);
            httpOut << body;
        }
        const char* bodyDelimiter = "\r\n\r\n";
        size_t bodyPos = str.find(bodyDelimiter);
        UNIT_ASSERT(bodyPos != TString::npos);
        return str.substr(bodyPos + strlen(bodyDelimiter));
    }

    TString SimulateBodyEncoding(const char* body) {
        TString bodyStr;
        TStringOutput bodyOut(bodyStr);
        TChunkedOutput chunkOut(&bodyOut);
        TZLibCompress comprOut(&chunkOut, ZLib::GZip);
        comprOut << body;
        return bodyStr;
    }

    Y_UNIT_TEST(TestRebuildStreamOnPost) {
        const char* body = "<html>Hello</html>";
        UNIT_ASSERT(MakeHttpOutputBody(body, false) == body);
        UNIT_ASSERT(MakeHttpOutputBody(body, true) == SimulateBodyEncoding(body));
    }

    Y_UNIT_TEST(TestOutputFinish) {
        TString str;
        TStringOutput strOut(str);
        TBufferedOutput bufOut(&strOut, 8192);
        THttpOutput httpOut(&bufOut);

        httpOut.EnableKeepAlive(true);
        httpOut.EnableCompression(true);

        const char* header = "GET / HTTP/1.1\r\nHost: yandex.ru\r\n\r\n";
        httpOut << header;

        unsigned curLen = str.size();
        const char* body = "<html>Hello</html>";
        httpOut << body;
        UNIT_ASSERT_VALUES_EQUAL(curLen, str.size());
        httpOut.Finish();
        UNIT_ASSERT_VALUES_EQUAL(curLen + strlen(body), str.size());
    }

    Y_UNIT_TEST(TestMultilineHeaders) {
        const char* headerLine0 = "HTTP/1.1 200 OK";
        const char* headerLine1 = "Content-Language: en";
        const char* headerLine2 = "Vary: Accept-Encoding, ";
        const char* headerLine3 = "\tAccept-Language";
        const char* headerLine4 = "Content-Length: 18";

        TString endLine("\r\n");
        TString r;
        r += headerLine0 + endLine;
        r += headerLine1 + endLine;
        r += headerLine2 + endLine;
        r += headerLine3 + endLine;
        r += headerLine4 + endLine + endLine;
        r += "<html>Hello</html>";
        TStringInput stringInput(r);
        THttpInput input(&stringInput);

        const THttpHeaders& httpHeaders = input.Headers();
        UNIT_ASSERT_VALUES_EQUAL(httpHeaders.Count(), 3u);

        THttpHeaders::TConstIterator it = httpHeaders.Begin();
        UNIT_ASSERT_VALUES_EQUAL(it->ToString(), TString(headerLine1));
        UNIT_ASSERT_VALUES_EQUAL((++it)->ToString(), TString::Join(headerLine2, headerLine3));
        UNIT_ASSERT_VALUES_EQUAL((++it)->ToString(), TString(headerLine4));
    }

    Y_UNIT_TEST(ContentLengthRemoval) {
        TMemoryInput request("GET / HTTP/1.1\r\nAccept-Encoding: gzip\r\n\r\n");
        THttpInput i(&request);
        TString result;
        TStringOutput out(result);
        THttpOutput httpOut(&out, &i);

        httpOut.EnableKeepAlive(true);
        httpOut.EnableCompression(true);
        httpOut << "HTTP/1.1 200 OK\r\n";
        char answer[] = "Mary had a little lamb.";
        httpOut << "Content-Length: " << strlen(answer) << "\r\n"
                                                           "\r\n";
        httpOut << answer;
        httpOut.Finish();

        Cdbg << result;
        result.to_lower();
        UNIT_ASSERT(result.Contains("content-encoding: gzip"));
        UNIT_ASSERT(!result.Contains("content-length"));
    }

    Y_UNIT_TEST(CodecsPriority) {
        TMemoryInput request("GET / HTTP/1.1\r\nAccept-Encoding: gzip, br\r\n\r\n");
        TVector<TStringBuf> codecs = {"br", "gzip"};

        THttpInput i(&request);
        TString result;
        TStringOutput out(result);
        THttpOutput httpOut(&out, &i);

        httpOut.EnableKeepAlive(true);
        httpOut.EnableCompression(codecs);
        httpOut << "HTTP/1.1 200 OK\r\n";
        char answer[] = "Mary had a little lamb.";
        httpOut << "Content-Length: " << strlen(answer) << "\r\n"
                                                           "\r\n";
        httpOut << answer;
        httpOut.Finish();

        Cdbg << result;
        result.to_lower();
        UNIT_ASSERT(result.Contains("content-encoding: br"));
    }

    Y_UNIT_TEST(CodecsPriority2) {
        TMemoryInput request("GET / HTTP/1.1\r\nAccept-Encoding: gzip, br\r\n\r\n");
        TVector<TStringBuf> codecs = {"gzip", "br"};

        THttpInput i(&request);
        TString result;
        TStringOutput out(result);
        THttpOutput httpOut(&out, &i);

        httpOut.EnableKeepAlive(true);
        httpOut.EnableCompression(codecs);
        httpOut << "HTTP/1.1 200 OK\r\n";
        char answer[] = "Mary had a little lamb.";
        httpOut << "Content-Length: " << strlen(answer) << "\r\n"
                                                           "\r\n";
        httpOut << answer;
        httpOut.Finish();

        Cdbg << result;
        result.to_lower();
        UNIT_ASSERT(result.Contains("content-encoding: gzip"));
    }

    Y_UNIT_TEST(HasTrailers) {
        TMemoryInput response(
            "HTTP/1.1 200 OK\r\n"
            "Transfer-Encoding: chunked\r\n"
            "\r\n"
            "3\r\n"
            "foo"
            "0\r\n"
            "Bar: baz\r\n"
            "\r\n");
        THttpInput i(&response);
        TMaybe<THttpHeaders> trailers = i.Trailers();
        UNIT_ASSERT(!trailers.Defined());
        i.ReadAll();
        trailers = i.Trailers();
        UNIT_ASSERT_VALUES_EQUAL(trailers.GetRef().Count(), 1);
        UNIT_ASSERT_VALUES_EQUAL(trailers.GetRef().Begin()->ToString(), "Bar: baz");
    }

    Y_UNIT_TEST(NoTrailersWithChunks) {
        TMemoryInput response(
            "HTTP/1.1 200 OK\r\n"
            "Transfer-Encoding: chunked\r\n"
            "\r\n"
            "3\r\n"
            "foo"
            "0\r\n"
            "\r\n");
        THttpInput i(&response);
        TMaybe<THttpHeaders> trailers = i.Trailers();
        UNIT_ASSERT(!trailers.Defined());
        i.ReadAll();
        trailers = i.Trailers();
        UNIT_ASSERT_VALUES_EQUAL(trailers.GetRef().Count(), 0);
    }

    Y_UNIT_TEST(NoTrailersNoChunks) {
        TMemoryInput response(
            "HTTP/1.1 200 OK\r\n"
            "Content-Length: 3\r\n"
            "\r\n"
            "bar");
        THttpInput i(&response);
        TMaybe<THttpHeaders> trailers = i.Trailers();
        UNIT_ASSERT(!trailers.Defined());
        i.ReadAll();
        trailers = i.Trailers();
        UNIT_ASSERT_VALUES_EQUAL(trailers.GetRef().Count(), 0);
    }

    Y_UNIT_TEST(RequestWithoutContentLength) {
        TStringStream request;
        {
            THttpOutput httpOutput(&request);
            httpOutput << "POST / HTTP/1.1\r\n"
                          "Host: yandex.ru\r\n"
                          "\r\n";
            httpOutput << "GGLOL";
        }
        {
            TStringInput input(request.Str());
            THttpInput httpInput(&input);
            bool chunkedOrHasContentLength = false;
            for (const auto& header : httpInput.Headers()) {
                if (header.Name() == "Transfer-Encoding" && header.Value() == "chunked" || header.Name() == "Content-Length") {
                    chunkedOrHasContentLength = true;
                }
            }

            // If request doesn't contain neither Content-Length header nor Transfer-Encoding header
            // then server considers message body length to be zero.
            // (See https://tools.ietf.org/html/rfc7230#section-3.3.3)
            UNIT_ASSERT(chunkedOrHasContentLength);

            UNIT_ASSERT_VALUES_EQUAL(httpInput.ReadAll(), "GGLOL");
        }
    }

    Y_UNIT_TEST(TestInputHasContent) {
        {
            TStringStream request;
            request << "POST / HTTP/1.1\r\n"
                       "Host: yandex.ru\r\n"
                       "\r\n";
            request << "HTTPDATA";

            TStringInput input(request.Str());
            THttpInput httpInput(&input);

            UNIT_ASSERT(!httpInput.HasContent());
            UNIT_ASSERT_VALUES_EQUAL(httpInput.ReadAll(), "");
        }

        {
            TStringStream request;
            request << "POST / HTTP/1.1\r\n"
                       "Host: yandex.ru\r\n"
                       "Content-Length: 8"
                       "\r\n\r\n";
            request << "HTTPDATA";

            TStringInput input(request.Str());
            THttpInput httpInput(&input);

            UNIT_ASSERT(httpInput.HasContent());
            UNIT_ASSERT_VALUES_EQUAL(httpInput.ReadAll(), "HTTPDATA");
        }

        {
            TStringStream request;
            request << "POST / HTTP/1.1\r\n"
                       "Host: yandex.ru\r\n"
                       "Transfer-Encoding: chunked"
                       "\r\n\r\n";
            request << "8\r\nHTTPDATA\r\n0\r\n";

            TStringInput input(request.Str());
            THttpInput httpInput(&input);

            UNIT_ASSERT(httpInput.HasContent());
            UNIT_ASSERT_VALUES_EQUAL(httpInput.ReadAll(), "HTTPDATA");
        }
    }

    Y_UNIT_TEST(TestHttpInputHeadRequest) {
        class THeadOnlyInput: public IInputStream {
        public:
            THeadOnlyInput() = default;

        private:
            size_t DoRead(void* buf, size_t len) override {
                if (Eof_) {
                    ythrow yexception() << "should not read after EOF";
                }

                const size_t toWrite = Min(len, Data_.size() - Pos_);
                if (toWrite == 0) {
                    Eof_ = true;
                    return 0;
                }

                memcpy(buf, Data_.data() + Pos_, toWrite);
                Pos_ += toWrite;
                return toWrite;
            }

        private:
            TString Data_{TStringBuf("HEAD / HTTP/1.1\r\nHost: yandex.ru\r\n\r\n")};
            size_t Pos_{0};
            bool Eof_{false};
        };
        THeadOnlyInput input;
        THttpInput httpInput(&input);

        UNIT_ASSERT(!httpInput.HasContent());
        UNIT_ASSERT_VALUES_EQUAL(httpInput.ReadAll(), "");
    }

    Y_UNIT_TEST(TestHttpOutputResponseToHeadRequestNoZeroChunk) {
        TStringStream request;
        request << "HEAD / HTTP/1.1\r\n"
                   "Host: yandex.ru\r\n"
                   "Connection: Keep-Alive\r\n"
                   "\r\n";

        TStringInput input(request.Str());
        THttpInput httpInput(&input);

        TStringStream outBuf;
        THttpOutput out(&outBuf, &httpInput);
        out.EnableKeepAlive(true);
        out << "HTTP/1.1 200 OK\r\nConnection: Keep-Alive\r\n\r\n";
        out << "";
        out.Finish();
        TString result = outBuf.Str();
        UNIT_ASSERT(!result.Contains(TStringBuf("0\r\n")));
    }

    Y_UNIT_TEST(TestHttpOutputDisableCompressionHeader) {
        TMemoryInput request("GET / HTTP/1.1\r\nAccept-Encoding: gzip\r\n\r\n");
        const TString data = "qqqqqqqqqqqqqqqqqqqqqqqqqqqqqq";

        THttpInput httpInput(&request);
        TString result;

        {
            TStringOutput output(result);
            THttpOutput httpOutput(&output, &httpInput);
            httpOutput.EnableCompressionHeader(false);
            httpOutput << "HTTP/1.1 200 OK\r\n"
                   "content-encoding: gzip\r\n"
                   "\r\n" + data;
            httpOutput.Finish();
        }

        UNIT_ASSERT(result.Contains("content-encoding: gzip"));
        UNIT_ASSERT(result.Contains(data));
    }

    size_t DoTestHttpOutputSize(const TString& res, bool enableCompession) {
        TTestHttpServer serverImpl(res);
        TPortManager pm;

        const ui16 port = pm.GetPort();
        THttpServer server(&serverImpl,
                           THttpServer::TOptions(port)
                                .EnableKeepAlive(true)
                                .EnableCompression(enableCompession));
        UNIT_ASSERT(server.Start());

        TNetworkAddress addr("localhost", port);
        TSocket s(addr);

        {
            TSocketOutput so(s);
            THttpOutput out(&so);
            out << "GET / HTTP/1.1\r\n"
                   "Host: www.yandex.ru\r\n"
                   "Connection: Keep-Alive\r\n"
                   "Accept-Encoding: gzip\r\n"
                   "\r\n";
            out.Finish();
        }

        TSocketInput si(s);
        THttpInput input(&si);

        unsigned httpCode = ParseHttpRetCode(input.FirstLine());
        UNIT_ASSERT_VALUES_EQUAL(httpCode, 200u);

        UNIT_ASSERT_VALUES_EQUAL(res, input.ReadAll());

        server.Stop();

        return serverImpl.LastRequestSentSize();
    }

    Y_UNIT_TEST(TestHttpOutputSize) {
        TString res = "qqqqqq";
        UNIT_ASSERT_VALUES_EQUAL(res.size(), DoTestHttpOutputSize(res, false));
        UNIT_ASSERT_VALUES_UNEQUAL(res.size(), DoTestHttpOutputSize(res, true));
    }
} // THttpStreamTest suite
