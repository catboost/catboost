#pragma once

#include <util/string/split.h>
#include <util/string/ascii.h>

namespace NNeh {
    namespace NHttp {
        inline auto SplitHeaders(TStringBuf headers) {
            return StringSplitter(headers).SplitByString("\r\n").SkipEmpty();
        }

        template <typename Port>
        void WriteHostHeader(IOutputStream& out, TStringBuf host, Port port) {
            out << AsStringBuf("Host: ") << host;
            if (port) {
                out << AsStringBuf(":") << port;
            }
            out << AsStringBuf("\r\n");
        }

        inline bool HasHostHeader(TStringBuf headers) {
            for (TStringBuf header : NNeh::NHttp::SplitHeaders(headers)) {
                if (AsciiHasPrefixIgnoreCase(header, "Host:")) {
                    return true;
                }
            }
            return false;
        }

        template <typename Port>
        void WriteHostHeaderIfNot(IOutputStream& out, TStringBuf host, Port port, TStringBuf headers) {
            if (!NNeh::NHttp::HasHostHeader(headers)) {
                NNeh::NHttp::WriteHostHeader(out, host, port);
            }
        }
    }
}
