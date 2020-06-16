#include "neh.h"
#include "http_common.h"
#include "http_headers.h"

#include <library/cpp/testing/unittest/registar.h>

Y_UNIT_TEST_SUITE(THttpCommon) {
    Y_UNIT_TEST(TCheckRequestFlags) {
        NNeh::TMessage msg = NNeh::TMessage::FromString("http://localhost:3380/ntables");
        UNIT_ASSERT(NNeh::NHttp::MakeFullRequest(msg, "", "", "", NNeh::NHttp::DefaultRequestType, NNeh::NHttp::ERequestFlag::AbsoluteUri));
        UNIT_ASSERT(msg.Data.StartsWith("GET http://localhost:3380/ntables HTTP/1.1"));
    }

    Y_UNIT_TEST(TMakeFullRequest) {
        {
            NNeh::TMessage msg = NNeh::TMessage::FromString("http://localhost:3380/ntables");
            UNIT_ASSERT(NNeh::NHttp::MakeFullRequest(msg, "", ""));
            UNIT_ASSERT_VALUES_EQUAL(msg.Addr, "full://localhost:3380/ntables");
            UNIT_ASSERT(msg.Data.StartsWith("GET /ntables HTTP/1.1"));
        }

        {
            NNeh::TMessage msg = NNeh::TMessage::FromString("https://localhost:3380/ntables");
            UNIT_ASSERT(NNeh::NHttp::MakeFullRequest(msg, "", ""));
            UNIT_ASSERT_VALUES_EQUAL(msg.Addr, "fulls://localhost:3380/ntables");
        }
    }

    Y_UNIT_TEST(TMakeFullRequestWithHost) {
        NNeh::TMessage msg = NNeh::TMessage::FromString("http://localhost:3380/ntables");
        const TString headers =
            "Accept: text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8\r\n"
            "Host: yandex.ru\r\n";

        UNIT_ASSERT(NNeh::NHttp::MakeFullRequest(msg, headers, ""));
        UNIT_ASSERT(msg.Data.Contains("Accept: text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8\r\n"));
        UNIT_ASSERT(msg.Data.Contains("Host: yandex.ru\r\n"));
        UNIT_ASSERT(!msg.Data.Contains("Host: localhost"));
    }

    Y_UNIT_TEST(TMakeFullRequestWithHost2) {
        NNeh::TMessage msg = NNeh::TMessage::FromString("http://localhost:3380/ntables");
        const TString headers =
            "Accept: text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8\r\n"
            "X-Some-Header: Host:yandex.ru\r\n";

        UNIT_ASSERT(NNeh::NHttp::MakeFullRequest(msg, headers, ""));
        UNIT_ASSERT(msg.Data.Contains("Accept: text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8\r\n"));
        UNIT_ASSERT(msg.Data.Contains("X-Some-Header: Host:yandex.ru\r\n"));
        UNIT_ASSERT(msg.Data.Contains("Host: localhost:3380\r\n"));
    }

    Y_UNIT_TEST(TMakeFullRequestWithHost3) {
        NNeh::TMessage msg = NNeh::TMessage::FromString("http://localhost:3380/ntables");
        const TString headers =
            "Accept: text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8\r\n"
            "X-Some-Header: Host:yandex.ru\r\n"
            "Host: yandex.ru";

        UNIT_ASSERT(NNeh::NHttp::MakeFullRequest(msg, headers, ""));
        UNIT_ASSERT(msg.Data.Contains("Accept: text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8\r\n"));
        UNIT_ASSERT(msg.Data.Contains("X-Some-Header: Host:yandex.ru\r\n"));
        UNIT_ASSERT(msg.Data.Contains("Host: yandex.ru\r\n"));
        UNIT_ASSERT(!msg.Data.Contains("Host: localhost"));
    }

    Y_UNIT_TEST(TMakeFullRequestWithContentLength1) {
        NNeh::TMessage msg = NNeh::TMessage::FromString("http://localhost:3380/ntables");
        const TString headers =
            "Accept: text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8\r\n"
            "Content-Length: 40\r\n";

        const TString content = "Some string 25 bytes long";

        UNIT_ASSERT(NNeh::NHttp::MakeFullRequest(msg, headers, content));
        UNIT_ASSERT(msg.Data.find("Content-Length: 25\r\n") != TString::npos);
        UNIT_ASSERT(msg.Data.find("Accept: text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8\r\n") != TString::npos);
        UNIT_ASSERT(msg.Data.find("Content-Length: 40") == TString::npos);
    }

    Y_UNIT_TEST(TMakeFullRequestWithContentLength2) {
        NNeh::TMessage msg = NNeh::TMessage::FromString("http://localhost:3380/ntables");
        const TString headers =
            "Content-Length: 40\r\n"
            "Accept: text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8\r\n";

        const TString content = "Some string 25 bytes long";

        UNIT_ASSERT(NNeh::NHttp::MakeFullRequest(msg, headers, content));
        UNIT_ASSERT(msg.Data.find("Content-Length: 25\r\n") != TString::npos);
        UNIT_ASSERT(msg.Data.find("Accept: text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8\r\n") != TString::npos);
        UNIT_ASSERT(msg.Data.find("Content-Length: 40") == TString::npos);
    }

    Y_UNIT_TEST(TMakeFullRequestWithContentLength3) {
        NNeh::TMessage msg = NNeh::TMessage::FromString("http://localhost:3380/ntables");
        const TString headers =
            "Accept: text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8\r\n"
            "Content-Length: 40\r\n"
            "Accept-Encoding: identity\r\n";

        const TString content = "Some string 25 bytes long";

        UNIT_ASSERT(NNeh::NHttp::MakeFullRequest(msg, headers, content));
        UNIT_ASSERT(msg.Data.find("Content-Length: 25\r\n") != TString::npos);
        UNIT_ASSERT(msg.Data.find("Accept: text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8\r\n") != TString::npos);
        UNIT_ASSERT(msg.Data.find("Accept-Encoding: identity\r\n") != TString::npos);
        UNIT_ASSERT(msg.Data.find("Content-Length: 40") == TString::npos);
    }

    Y_UNIT_TEST(TMakeFullRequestWithContentLength4) {
        NNeh::TMessage msg = NNeh::TMessage::FromString("http://localhost:3380/ntables");
        const TString headers =
            "Accept: text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8\r\n"
            "Accept-Encoding: identity\r\n";

        const TString content = "Some string 25 bytes long";

        UNIT_ASSERT(NNeh::NHttp::MakeFullRequest(msg, headers, content));
        UNIT_ASSERT(msg.Data.find("Content-Length: 25\r\n") != TString::npos);
        UNIT_ASSERT(msg.Data.find("Accept: text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8\r\n") != TString::npos);
        UNIT_ASSERT(msg.Data.find("Accept-Encoding: identity\r\n") != TString::npos);
        UNIT_ASSERT(msg.Data.find("Content-Length: 40") == TString::npos);
    }

    Y_UNIT_TEST(TMakeFullRequestWithContentLength5) {
        NNeh::TMessage msg = NNeh::TMessage::FromString("http://localhost:3380/ntables");
        const TString headers =
            "Accept: text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8\r\n"
            "Content-Length: 40\r\n"
            "Accept-Encoding: identity\r\n";

        UNIT_ASSERT(NNeh::NHttp::MakeFullRequest(msg, headers, ""));
        UNIT_ASSERT(msg.Data.find("Accept: text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8\r\n") != TString::npos);
        UNIT_ASSERT(msg.Data.find("Accept-Encoding: identity\r\n") != TString::npos);
        UNIT_ASSERT(msg.Data.find("Content-Length: 25") == TString::npos);
        UNIT_ASSERT(msg.Data.find("Content-Length: 40") == TString::npos);
    }

    Y_UNIT_TEST(TMakeFullRequestWithContentLengthCaseInsensitive) {
        NNeh::TMessage msg = NNeh::TMessage::FromString("http://localhost:3380/ntables");
        const TString headers =
            "Accept: text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8\r\n"
            "content-length: 40\r\n"
            "content-Length: 40\r\n"
            "Content-length: 40\r\n"
            "Accept-Encoding: identity\r\n";

        const TString content = "Some string 25 bytes long";

        UNIT_ASSERT(NNeh::NHttp::MakeFullRequest(msg, headers, content));
        UNIT_ASSERT(msg.Data.find("Content-Length: 25\r\n") != TString::npos);
        UNIT_ASSERT(msg.Data.find("Accept: text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8\r\n") != TString::npos);
        UNIT_ASSERT(msg.Data.find("Accept-Encoding: identity\r\n") != TString::npos);
        UNIT_ASSERT(msg.Data.find("Content-Length: 40") == TString::npos);
        UNIT_ASSERT(msg.Data.find("content-length: 40") == TString::npos);
        UNIT_ASSERT(msg.Data.find("Content-length: 40") == TString::npos);
        UNIT_ASSERT(msg.Data.find("content-Length: 40") == TString::npos);
    }

    Y_UNIT_TEST(TMakeFullRequest1) {
        /// Test for preserving behaviour.
        NNeh::TMessage msg = NNeh::TMessage::FromString("post://localhost:3380/ntables");
        const TString headers =
            "Accept: text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8\r\n"
            "Content-Length: 40\r\n"
            "Accept-Encoding: identity\r\n";
        const TString content = "Some string 25 bytes long";
        const TString contentType = "text/html; charset=utf-8";
        const TVector<TString> urlParts = {TString("text=query"), TString("lr=213")};

        UNIT_ASSERT(!NNeh::NHttp::MakeFullRequest(msg, urlParts, headers, content, contentType));
    }

    Y_UNIT_TEST(TMakeFullRequest2) {
        /// Test for preserving behaviour.
        NNeh::TMessage msg = NNeh::TMessage::FromString("full://localhost:3380/ntables");
        const TString headers =
            "Accept: text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8\r\n"
            "Content-Length: 40\r\n"
            "Accept-Encoding: identity\r\n";
        const TString content = "Some string 25 bytes long";
        const TString contentType = "text/html; charset=utf-8";

        UNIT_ASSERT(!NNeh::NHttp::MakeFullRequest(msg, headers, content, contentType));
    }

    Y_UNIT_TEST(TMakeFullRequest3) {
        /// Test for preserving behaviour.
        NNeh::TMessage msg = NNeh::TMessage::FromString("full://localhost:3380/ntables");
        const TString headers =
            "Accept: text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8\r\n"
            "Content-Length: 40\r\n"
            "Accept-Encoding: identity\r\n";
        const TString contentType = "text/html; charset=utf-8";
        const TVector<TString> urlParts = {TString("text=query"), TString("lr=213")};

        UNIT_ASSERT(!NNeh::NHttp::MakeFullRequest(msg, urlParts, headers, "", contentType));
    }

    Y_UNIT_TEST(TMakeFullRequestPost1) {
        /// Test for preserving behaviour.
        NNeh::TMessage msg = NNeh::TMessage::FromString("http://localhost:3380/ntables");
        const TString headers =
            "Accept: text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8\r\n"
            "Content-Length: 40\r\n"
            "Accept-Encoding: identity\r\n";
        const TString content = "Some string 25 bytes long";
        const TString contentType = "text/html; charset=utf-8";

        UNIT_ASSERT(NNeh::NHttp::MakeFullRequest(msg, headers, content, contentType));

        UNIT_ASSERT_EQUAL_C(msg.Data,
                            "POST /ntables HTTP/1.1\r\n"
                            "Host: localhost:3380\r\n"
                            "Accept: text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8\r\n"
                            "Accept-Encoding: identity\r\n"
                            "Content-Type: text/html; charset=utf-8\r\n"
                            "Content-Length: 25\r\n"
                            "\r\n"
                            "Some string 25 bytes long",
                            msg.Data);
    }

    Y_UNIT_TEST(TMakeFullRequestPost2) {
        /// Test for preserving behaviour.
        NNeh::TMessage msg = NNeh::TMessage::FromString("post://localhost:3380/ntables");
        const TString headers =
            "Accept: text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8\r\n"
            "Accept-Encoding: identity\r\n";
        const TString contentType = "text/html; charset=utf-8";
        const TVector<TString> urlParts = {TString("text=query"), TString("lr=213")};

        UNIT_ASSERT(NNeh::NHttp::MakeFullRequest(msg, urlParts, headers, "", contentType));
        UNIT_ASSERT_EQUAL_C(msg.Data,
                            "POST /ntables HTTP/1.1\r\n"
                            "Host: localhost:3380\r\n"
                            "Accept: text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8\r\n"
                            "Accept-Encoding: identity\r\n"
                            "Content-Type: text/html; charset=utf-8\r\n"
                            "Content-Length: 17\r\n"
                            "\r\n"
                            "text=query&lr=213",
                            msg.Data);
    }

    Y_UNIT_TEST(TMakeFullRequestPost3) {
        /// Test for preserving behaviour.
        NNeh::TMessage msg = NNeh::TMessage::FromString("http://localhost:3380/ntables");
        const TString headers =
            "Accept: text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8\r\n"
            "Accept-Encoding: identity\r\n";
        const TString content = "Some string 25 bytes long";
        const TString contentType = "text/html; charset=utf-8";
        const TVector<TString> urlParts = {TString("text=query"), TString("lr=213")};

        UNIT_ASSERT(NNeh::NHttp::MakeFullRequest(msg, urlParts, headers, content, contentType));
        UNIT_ASSERT_EQUAL_C(msg.Data,
                            "POST /ntables?text=query&lr=213 HTTP/1.1\r\n"
                            "Host: localhost:3380\r\n"
                            "Accept: text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8\r\n"
                            "Accept-Encoding: identity\r\n"
                            "Content-Type: text/html; charset=utf-8\r\n"
                            "Content-Length: 25\r\n"
                            "\r\n"
                            "Some string 25 bytes long",
                            msg.Data);
    }

    Y_UNIT_TEST(TMakeFullRequestGet1) {
        /// Test for preserving behaviour.
        NNeh::TMessage msg = NNeh::TMessage::FromString("http://localhost:3380/ntables");
        const TString headers =
            "Accept: text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8\r\n"
            "Accept-Encoding: identity\r\n";
        const TVector<TString> urlParts = {TString("text=query"), TString("lr=213")};

        UNIT_ASSERT(NNeh::NHttp::MakeFullRequest(msg, urlParts, headers, "", ""));
        UNIT_ASSERT_EQUAL_C(msg.Data,
                            "GET /ntables?text=query&lr=213 HTTP/1.1\r\n"
                            "Host: localhost:3380\r\n"
                            "Accept: text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8\r\n"
                            "Accept-Encoding: identity\r\n"
                            "\r\n",
                            msg.Data);
    }

    Y_UNIT_TEST(TMakeFullRequestBrokenHeaderSplitter) {
        /// Test for preserving behaviour.
        NNeh::TMessage msg = NNeh::TMessage::FromString("http://localhost:3380/ntables");
        const TString headers =
            "Accept: text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8\n"
            "Accept-Encoding: identity\n\r\n";
        const TVector<TString> urlParts = {TString("text=query"), TString("lr=213")};

        UNIT_ASSERT(NNeh::NHttp::MakeFullRequest(msg, urlParts, headers, "", ""));
        UNIT_ASSERT_EQUAL_C(msg.Data,
                            "GET /ntables?text=query&lr=213 HTTP/1.1\r\n"
                            "Host: localhost:3380\r\n"
                            "Accept: text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8\r\n"
                            "Accept-Encoding: identity\r\n"
                            "\r\n",
                            msg.Data);
    }

    Y_UNIT_TEST(TMakeFullRequestPut1) {
        /// Test for preserving behaviour.
        using NNeh::NHttp::ERequestType;
        NNeh::TMessage msg = NNeh::TMessage::FromString("http://localhost:3380/ntables");
        const TString headers =
            "Accept: text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8\r\n"
            "Accept-Encoding: identity\r\n";
        const TString content = "Some string 25 bytes long";
        const TString contentType = "text/html; charset=utf-8";
        const TVector<TString> urlParts = {TString("text=query"), TString("lr=213")};

        UNIT_ASSERT(NNeh::NHttp::MakeFullRequest(msg, urlParts, headers, content, contentType, ERequestType::Put));
        UNIT_ASSERT_EQUAL_C(msg.Data,
                            "PUT /ntables?text=query&lr=213 HTTP/1.1\r\n"
                            "Host: localhost:3380\r\n"
                            "Accept: text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8\r\n"
                            "Accept-Encoding: identity\r\n"
                            "Content-Type: text/html; charset=utf-8\r\n"
                            "Content-Length: 25\r\n"
                            "\r\n"
                            "Some string 25 bytes long",
                            msg.Data);
    }

    Y_UNIT_TEST(TMakeFullRequestDelete1) {
        /// Test for preserving behaviour.
        using NNeh::NHttp::ERequestType;
        NNeh::TMessage msg = NNeh::TMessage::FromString("http://localhost:3380/ntables");
        const TString headers =
            "Accept: text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8\r\n"
            "Accept-Encoding: identity\r\n";
        const TVector<TString> urlParts = {TString("text=query"), TString("lr=213")};

        UNIT_ASSERT(NNeh::NHttp::MakeFullRequest(msg, urlParts, headers, "", "", ERequestType::Delete));
        UNIT_ASSERT_EQUAL_C(msg.Data,
                            "DELETE /ntables?text=query&lr=213 HTTP/1.1\r\n"
                            "Host: localhost:3380\r\n"
                            "Accept: text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8\r\n"
                            "Accept-Encoding: identity\r\n"
                            "\r\n",
                            msg.Data);
    }

    Y_UNIT_TEST(TMakeFullRequestPatch1) {
        /// Test for preserving behaviour.
        using NNeh::NHttp::ERequestType;
        NNeh::TMessage msg = NNeh::TMessage::FromString("http://localhost:3380/ntables/123");
        const TStringBuf content = "{\"key\":\"newValue\"}";
        const TString contentType = "application/json";

        UNIT_ASSERT(NNeh::NHttp::MakeFullRequest(msg, {}, {}, content, contentType, ERequestType::Patch));
        UNIT_ASSERT_EQUAL_C(msg.Data,
                            "PATCH /ntables/123 HTTP/1.1\r\n"
                            "Host: localhost:3380\r\n"
                            "Content-Type: application/json\r\n"
                            "Content-Length: 18\r\n"
                            "\r\n"
                            "{\"key\":\"newValue\"}",
                            msg.Data);
    }

    Y_UNIT_TEST(TConvertRequestTypeToString) {
        UNIT_ASSERT_VALUES_EQUAL(ToString(NNeh::NHttp::ERequestType::Get), "GET");

        TStringStream ss;
        ss << NNeh::NHttp::ERequestType::Post;
        UNIT_ASSERT_VALUES_EQUAL(ss.Str(), "POST");
    }

    Y_UNIT_TEST(TIsHttpScheme) {
        UNIT_ASSERT(NNeh::NHttp::IsHttpScheme("http"));
        UNIT_ASSERT(NNeh::NHttp::IsHttpScheme("post"));
        UNIT_ASSERT(!NNeh::NHttp::IsHttpScheme("inproc"));
    }

    Y_UNIT_TEST(TSplitHeaders) {
        TString headers = "Host: yandex.ru\r\nContent-Length: 18\r\nX-Header: v:5\n\r\nX-Header2: yyy\nX-Header3: zzz";
        TVector<TString> expected = {"Host: yandex.ru", "Content-Length: 18", "X-Header: v:5", "X-Header2: yyy", "X-Header3: zzz"};

        size_t i = 0;
        NNeh::NHttp::THeaderSplitter splitter{headers};

        TStringBuf header;
        while (splitter.Next(header)) {
            UNIT_ASSERT_VALUES_EQUAL(header, expected[i++]);
        };

        UNIT_ASSERT_VALUES_EQUAL(i, expected.size());
    }


    Y_UNIT_TEST(TMakeFullProxyRequest) {
        {
            NNeh::TMessage msg = NNeh::TMessage::FromString("http://localhost:3380/ntables");
            UNIT_ASSERT(NNeh::NHttp::MakeFullProxyRequest(msg, "http://proxyhost:4242", "", ""));
            UNIT_ASSERT_VALUES_EQUAL(msg.Addr, "full://proxyhost:4242");
            UNIT_ASSERT(msg.Data.StartsWith("GET http://localhost:3380/ntables HTTP/1.1"));
        }

        {
            NNeh::TMessage msg = NNeh::TMessage::FromString("https://localhost:3380/ntables");
            UNIT_ASSERT(NNeh::NHttp::MakeFullProxyRequest(msg, "http://proxyhost:4242", "", ""));
            UNIT_ASSERT_VALUES_EQUAL(msg.Addr, "full://proxyhost:4242");
            UNIT_ASSERT(msg.Data.StartsWith("GET https://localhost:3380/ntables HTTP/1.1"));
        }
    }
}
