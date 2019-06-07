#include "http_common.h"

#include "location.h"
#include "http_headers.h"

#include <util/stream/str.h>
#include <util/stream/length.h>
#include <util/stream/null.h>
#include <util/string/ascii.h>
#include <util/generic/singleton.h>

using NNeh::NHttp::ERequestType;

namespace {
    bool IsEmpty(const TStringBuf url) {
        return url.empty();
    }

    void WriteImpl(const TStringBuf url, IOutputStream& out) {
        out << url;
    }

    bool IsEmpty(const TVector<TString>& urlParts) {
        return urlParts.empty();
    }

    void WriteImpl(const TVector<TString>& urlParts, IOutputStream& out) {
        NNeh::NHttp::JoinUrlParts(urlParts, out);
    }

    template <typename T>
    size_t GetLength(const T& urlParts) {
        TCountingOutput out(&Cnull);
        WriteImpl(urlParts, out);
        return out.Counter();
    }

    template <typename T>
    void WriteUrl(const T& urlParts, IOutputStream& out) {
        if (!IsEmpty(urlParts)) {
            out << '?';
            WriteImpl(urlParts, out);
        }
    }
}

namespace NNeh {
    namespace NHttp {
        size_t GetUrlPartsLength(const TVector<TString>& urlParts) {
            size_t res = 0;

            for (const TString& u : urlParts) {
                res += u.length();
            }

            if (urlParts.size() > 0) {
                res += urlParts.size() - 1; //'&' between parts
            }

            return res;
        }

        void JoinUrlParts(const TVector<TString>& urlParts, IOutputStream& out) {
            if (urlParts.empty()) {
                return;
            }

            out << urlParts[0];

            for (size_t i = 1; i < urlParts.size(); ++i) {
                out << '&' << urlParts[i];
            }
        }

        void WriteUrlParts(const TVector<TString>& urlParts, IOutputStream& out) {
            WriteUrl(urlParts, out);
        }
    }
}

namespace {
    const TStringBuf schemeHttps = "https";
    const TStringBuf schemeHttp = "http";
    const TStringBuf schemeHttp2 = "http2";
    const TStringBuf schemePost = "post";
    const TStringBuf schemePosts = "posts";
    const TStringBuf schemePost2 = "post2";
    const TStringBuf schemeFull = "full";
    const TStringBuf schemeFulls = "fulls";

    /*
        @brief  SafeWriteHeaders    write headers from hdrs to out with some checks:
                    - filter out Content-Lenthgh because we'll add it ourselfs later.

        @todo ensure headers right formatted (now receive from perl report bad format headers)
     */
    void SafeWriteHeaders(IOutputStream& out, TStringBuf hdrs) {
        NNeh::NHttp::THeaderSplitter splitter(hdrs);
        TStringBuf msgHdr;
        while (splitter.Next(msgHdr)) {
            if (!AsciiHasPrefixIgnoreCase(msgHdr, AsStringBuf("Content-Length"))) {
                out << msgHdr << AsStringBuf("\r\n");
            }
        }
    }

    template <typename T, typename W>
    TString BuildRequest(const NNeh::TParsedLocation& loc, const T& urlParams, const TStringBuf headers, const W& content, const TStringBuf contentType, ERequestType requestType, NNeh::NHttp::ERequestFlags requestFlags) {
        const bool isAbsoluteUri = requestFlags.HasFlags(NNeh::NHttp::ERequestFlag::AbsoluteUri);

        const auto contentLength = GetLength(content);
        TStringStream out;
        out.Reserve(loc.Service.length() + loc.Host.length() + GetLength(urlParams) + headers.length() + contentType.length() + contentLength + (isAbsoluteUri ? (loc.Host.length() + 13) : 0) // 13 - is a max port number length + scheme length
                    + 96);                                                                                                                                                                     //just some extra space

        Y_ASSERT(requestType != ERequestType::Any);
        out << requestType;
        out << ' ';
        if (isAbsoluteUri) {
            out << loc.Scheme << AsStringBuf("://") << loc.Host;
            if (loc.Port) {
                out << ':' << loc.Port;
            }
        }
        out << '/' << loc.Service;

        WriteUrl(urlParams, out);
        out << AsStringBuf(" HTTP/1.1\r\n");

        NNeh::NHttp::WriteHostHeaderIfNot(out, loc.Host, loc.Port, headers);
        SafeWriteHeaders(out, headers);
        if (!IsEmpty(content)) {
            if (!!contentType && headers.find(AsStringBuf("Content-Type:")) == TString::npos) {
                out << AsStringBuf("Content-Type: ") << contentType << AsStringBuf("\r\n");
            }
            out << AsStringBuf("Content-Length: ") << contentLength << AsStringBuf("\r\n");
            out << AsStringBuf("\r\n");
            WriteImpl(content, out);
        } else {
            out << AsStringBuf("\r\n");
        }
        return out.Str();
    }

    bool NeedGetRequestFor(TStringBuf scheme) {
        return scheme == schemeHttp2 || scheme == schemeHttp || scheme == schemeHttps;
    }

    bool NeedPostRequestFor(TStringBuf scheme) {
        return scheme == schemePost2 || scheme == schemePost || scheme == schemePosts;
    }

    inline ERequestType ChooseReqType(ERequestType userReqType, ERequestType defaultReqType) {
        Y_ASSERT(defaultReqType != ERequestType::Any);
        return userReqType != ERequestType::Any ? userReqType : defaultReqType;
    }
}

namespace NNeh {
    namespace NHttp {
        const TString DefaultContentType = "application/x-www-form-urlencoded";

        template <typename T>
        bool MakeFullRequestImpl(TMessage& msg, const T& urlParams, const TStringBuf headers, const TStringBuf content, const TStringBuf contentType, ERequestType reqType, ERequestFlags reqFlags) {
            const NNeh::TParsedLocation loc(msg.Addr);

            if (content.size()) {
                //content MUST be placed inside POST requests
                if (!IsEmpty(urlParams)) {
                    if (NeedGetRequestFor(loc.Scheme)) {
                        msg.Data = BuildRequest(loc, urlParams, headers, content, contentType, ChooseReqType(reqType, ERequestType::Post), reqFlags);
                    } else {
                        // cannot place in first header line potentially unsafe data from POST message
                        // (can contain forbidden for url-path characters)
                        // so support such mutation only for GET requests
                        return false;
                    }
                } else {
                    if (NeedGetRequestFor(loc.Scheme) || NeedPostRequestFor(loc.Scheme)) {
                        msg.Data = BuildRequest(loc, urlParams, headers, content, contentType, ChooseReqType(reqType, ERequestType::Post), reqFlags);
                    } else {
                        return false;
                    }
                }
            } else {
                if (NeedGetRequestFor(loc.Scheme)) {
                    msg.Data = BuildRequest(loc, urlParams, headers, "", "", ChooseReqType(reqType, ERequestType::Get), reqFlags);
                } else if (NeedPostRequestFor(loc.Scheme)) {
                    msg.Data = BuildRequest(loc, TString(), headers, urlParams, contentType, ChooseReqType(reqType, ERequestType::Post), reqFlags);
                } else {
                    return false;
                }
            }

            // ugly but still... https2 will break it :(
            if ('s' == loc.Scheme[loc.Scheme.size() - 1]) {
                msg.Addr.replace(0, schemeFulls.size(), schemeFulls);
            } else {
                msg.Addr.replace(0, schemeFull.size(), schemeFull);
            }

            return true;
        }

        bool MakeFullRequest(TMessage& msg, const TStringBuf headers, const TStringBuf content, const TStringBuf contentType, ERequestType reqType, ERequestFlags reqFlags) {
            return MakeFullRequestImpl(msg, msg.Data, headers, content, contentType, reqType, reqFlags);
        }

        bool MakeFullRequest(TMessage& msg, const TVector<TString>& urlParts, const TStringBuf headers, const TStringBuf content, const TStringBuf contentType, ERequestType reqType, ERequestFlags reqFlags) {
            return MakeFullRequestImpl(msg, urlParts, headers, content, contentType, reqType, reqFlags);
        }

        bool IsHttpScheme(TStringBuf scheme) {
            return NeedGetRequestFor(scheme) || NeedPostRequestFor(scheme);
        }
    }
}

